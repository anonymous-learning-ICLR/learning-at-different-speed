"""Paper reconstruction, not a historical-checkpoint-compatible replacement.

Experts retain the released recurrent decoders. Their 4-D outputs are the expert
representation; H_track/H_intensity are fixed coordinate projections. Thus mixing
before these linear heads is exactly Eq. (8). No best-of-K or sampled expert path.
"""
import copy
import torch
from TCNM.models_prior_unet4to4_tsi_router import TrajectoryGenerator as LegacyGenerator


class PaperGenerator(LegacyGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Independent learned task routers, initialized identically for all ablations.
        self.intensity_router = copy.deepcopy(self.net_chooser)
        self.register_buffer('task_alpha', torch.ones(2))
        self.register_buffer('gap_penalty', torch.zeros(2, self.num_gs))
        self.register_buffer('gap_gamma', torch.tensor(0.0))
        self.register_buffer('route_tau', torch.tensor(1.0))

    @torch.no_grad()
    def set_feedback(self, alpha, penalty, gamma, tau_r):
        a = torch.as_tensor(alpha, device=self.task_alpha.device, dtype=self.task_alpha.dtype).detach()
        p = torch.as_tensor(penalty, device=self.gap_penalty.device, dtype=self.gap_penalty.dtype).detach()
        if a.shape != self.task_alpha.shape or p.shape != self.gap_penalty.shape:
            raise ValueError('Feedback dimensions do not match model')
        if not torch.isfinite(a).all() or not torch.isfinite(p).all() or (a <= 0).any() or (p < 0).any():
            raise ValueError('Invalid feedback')
        if abs(float(a.sum()) - 2) > 1e-5 or gamma < 0 or tau_r <= 0:
            raise ValueError('Invalid normalization or routing constants')
        self.task_alpha.copy_(a)
        self.gap_penalty.copy_(p)
        self.gap_gamma.fill_(gamma)
        self.route_tau.fill_(tau_r)

    def route_weights(self, features):
        logits = torch.stack([self.net_chooser(features), self.intensity_router(features)], dim=1)
        corrected = (logits - self.gap_gamma * self.gap_penalty.detach()) / self.route_tau
        return torch.softmax(corrected, dim=-1), logits

    def forward(
        self,
        obs_traj,
        obs_traj_rel,
        seq_start_end,
        image_obs,
        env_data,
        user_noise=None,
    ):
        """Return differentiable mixtures AND individual expert predictions."""
        device = obs_traj.device
        batch = obs_traj_rel.size(1)
        obs_len = obs_traj_rel.size(0)

        image_obs_perm = image_obs.permute(0, 2, 1, 3, 4).contiguous()
        img_input_real = image_obs_perm.view(batch, obs_len, -1)
        encoder_img_real = self.img_embedding_real(img_input_real).permute(1, 0, 2)

        final_encoder_env = self.encoder_env(obs_traj_rel, encoder_img_real)
        h_n = final_encoder_env['final_h'][0]
        final_encoder_env_h = h_n[-1] if h_n.dim() == 3 else h_n

        env_feature_chooser, _, _ = self.env_net_chooser(env_data, image_obs)
        assert final_encoder_env_h.size(0) == env_feature_chooser.size(0)

        dec_h_evn = self.feature2dech_env(torch.cat([final_encoder_env_h, env_feature_chooser], dim=1))

        predrnn_out = self.Unet(image_obs)

        first_img = image_obs[:, :, 0].unsqueeze(2)
        all_img = torch.cat([first_img, predrnn_out], dim=2)

        img_input = all_img.view(batch, self.obs_len + self.pred_len, -1)
        img_embed_input = self.img_embedding(img_input).permute(1, 0, 2)

        encoder_img = img_embed_input[: self.obs_len]
        final_encoder = self.encoder(obs_traj_rel, encoder_img)
        h_gen = final_encoder['final_h'][0]
        final_encoder_h = h_gen[-1] if h_gen.dim() == 3 else h_gen

        dec_h = self.feature2dech(torch.cat([final_encoder_h, env_feature_chooser], dim=1))
        image_out = all_img

        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        decoder_img = img_embed_input[obs_len:]
        last_img = img_embed_input[obs_len - 1]

        # All experts are evaluated with the same latent noise, if enabled.
        state_tuple = self.mix_noise(dec_h.unsqueeze(0), seq_start_end, batch, user_noise=user_noise)
        outputs = []
        for expert in self.gs:
            out, _ = expert(obs_traj, obs_traj_rel, last_pos, last_pos_rel,
                            state_tuple, seq_start_end, decoder_img, last_img)
            outputs.append(out)
        experts = torch.stack(outputs, dim=1)  # [horizon, expert, batch, 4]
        weights, logits = self.route_weights(dec_h_evn)  # [batch, task, expert]
        track = torch.einsum('b k,p k b d->p b d', weights[:, 0], experts[..., :2])
        intensity = torch.einsum('b k,p k b d->p b d', weights[:, 1], experts[..., 2:])
        return dict(relative=torch.cat([track, intensity], dim=-1),
                    experts_relative=experts, weights=weights, logits=logits, image=image_out)
