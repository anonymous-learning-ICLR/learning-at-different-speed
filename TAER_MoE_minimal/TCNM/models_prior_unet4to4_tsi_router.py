import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from TCNM.Unet3D_merge_tiny4to4_dynamic import Unet3D
from TCNM.env_net_transformer_gphsplit import Env_net


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type, device):
    if noise_type == 'gaussian':
        return torch.randn(*shape, device=device)
    if noise_type == 'uniform':
        return (torch.rand(*shape, device=device) - 0.5) * 2.0
    raise ValueError(f'Unrecognized noise type "{noise_type}"')


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and TrajectoryDiscriminator."""

    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super().__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(4, embedding_dim)
        self.time_embedding = nn.Linear(4, embedding_dim)

    def init_hidden(self, batch, device):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim, device=device),
            torch.zeros(self.num_layers, batch, self.h_dim, device=device),
        )

    def forward(self, obs_traj, img_embed_input):
        """
        Inputs:
        - obs_traj: (obs_len, batch, 4)
        - img_embed_input: (obs_len, batch, embedding_dim)
        Output:
        - dict with keys 'final_h' and 'output'
        """
        device = obs_traj.device
        batch = obs_traj.size(1)
        input_dim = obs_traj.size(2)

        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, input_dim))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        obs_traj_embedding = obs_traj_embedding + img_embed_input

        state_tuple = self.init_hidden(batch, device)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        return {'final_h': state, 'output': output}


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator."""

    def __init__(
        self,
        seq_len,
        embedding_dim=64,
        h_dim=128,
        mlp_dim=1024,
        num_layers=1,
        pool_every_timestep=True,
        dropout=0.0,
        bottleneck_dim=1024,
        activation='relu',
        batch_norm=True,
        pooling_type='pool_net',
        neighborhood_size=2.0,
        grid_size=8,
        embeddings_dim=128,
        h_dims=128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep
        self.num_layers = num_layers

        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        if pool_every_timestep:
            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )

        self.spatial_embedding = nn.Linear(4, embedding_dim)
        self.time_embedding = nn.Linear(4, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 4)

    def init_hidden(self, batch, device):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim, device=device),
            torch.zeros(self.num_layers, batch, self.h_dim, device=device),
        )

    def forward(
        self,
        obs_traj,
        obs_traj_rel,
        last_pos,
        last_pos_rel,
        state_tuple,
        seq_start_end,
        decoder_img,
        last_img,
    ):
        """
        Inputs:
        - last_pos: (batch, 4)
        - last_pos_rel: (batch, 4)
        - state_tuple: (hh, ch) each (num_layers, batch, h_dim)
        - decoder_img: (pred_len, batch, embedding_dim)
        - last_img: (batch, embedding_dim)
        Output:
        - pred_traj_fake_rel: (pred_len, batch, 4)
        """
        batch = last_pos.size(0)

        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)
        decoder_input = decoder_input + last_img.unsqueeze(0)

        for i_step in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            rel_pos_step = rel_pos.unsqueeze(0)
            decoder_input = self.spatial_embedding(rel_pos_step)
            decoder_img_one = decoder_img[i_step].unsqueeze(0)
            decoder_input = decoder_input + decoder_img_one

            pred_traj_fake_rel.append(rel_pos_step.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class TrajectoryGenerator(nn.Module):
    """
    Generator with optional TSI-aware router control.

    The default behavior is backward compatible with the original file.
    To enable TSI-aware routing during training, call:
        generator.set_tsi_control(lambda_c, track_score, intensity_score)
    before the forward pass.

    The router then applies:
    1) a scale-aware bias over experts (small-scale <-> large-scale)
    2) a temperature adjustment on router logits
    """

    def __init__(
        self,
        obs_len,
        pred_len,
        embedding_dim=64,
        encoder_h_dim=64,
        decoder_h_dim=128,
        mlp_dim=1024,
        num_layers=1,
        noise_dim=(0,),
        noise_type='gaussian',
        noise_mix_type='ped',
        pooling_type=None,
        pool_every_timestep=True,
        dropout=0.0,
        bottleneck_dim=1024,
        activation='relu',
        batch_norm=True,
        neighborhood_size=2.0,
        grid_size=8,
        num_gs=6,
        num_sample=6,
        router_bias_strength=1.0,
        router_temp_kappa=0.35,
        router_temp_min=0.75,
        router_temp_max=1.25,
    ):
        super().__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.seq_len = self.obs_len + self.pred_len

        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = bottleneck_dim
        self.num_gs = num_gs
        self.num_sample = num_sample

        self.router_bias_strength = float(router_bias_strength)
        self.router_temp_kappa = float(router_temp_kappa)
        self.router_temp_min = float(router_temp_min)
        self.router_temp_max = float(router_temp_max)

        self.Unet = Unet3D(1, 1, obs_len=self.obs_len, pred_len=self.pred_len)

        self.img_embedding = nn.Linear(64 * 64, self.embedding_dim)
        self.img_embedding_real = nn.Linear(64 * 64, self.embedding_dim)

        self.env_net = Env_net(obs_len=self.obs_len)
        self.env_net_chooser = Env_net(obs_len=self.obs_len)

        self.feature2dech_env = nn.Linear(128, encoder_h_dim)
        self.feature2dech = nn.Linear(128, encoder_h_dim)

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.encoder_env = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.gs = nn.ModuleList()
        for _ in range(num_gs):
            self.gs.append(
                Decoder(
                    pred_len,
                    embedding_dim=embedding_dim,
                    h_dim=decoder_h_dim,
                    mlp_dim=mlp_dim,
                    num_layers=num_layers,
                    pool_every_timestep=pool_every_timestep,
                    dropout=dropout,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    pooling_type=pooling_type,
                    grid_size=grid_size,
                    neighborhood_size=neighborhood_size,
                    embeddings_dim=embedding_dim,
                    h_dims=encoder_h_dim,
                )
            )

        self.net_chooser = nn.Sequential(
            nn.Linear(encoder_h_dim, encoder_h_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_h_dim // 2, encoder_h_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_h_dim // 2, num_gs),
        )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim]
            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )

        # Small-index experts are treated as smaller-scale / track-oriented,
        # large-index experts as larger-scale / intensity-oriented.
        self.register_buffer('expert_scale_prior', torch.linspace(-1.0, 1.0, self.num_gs))
        self.register_buffer('tsi_lambda_buf', torch.tensor(0.0))
        self.register_buffer('tsi_track_score_buf', torch.tensor(0.0))
        self.register_buffer('tsi_intensity_score_buf', torch.tensor(0.0))
        self.register_buffer('router_temperature_buf', torch.tensor(1.0))

    # ---------------------------------------------------------------------
    # TSI-aware router control API
    # ---------------------------------------------------------------------
    def clear_tsi_control(self):
        self.tsi_lambda_buf.fill_(0.0)
        self.tsi_track_score_buf.fill_(0.0)
        self.tsi_intensity_score_buf.fill_(0.0)
        self.router_temperature_buf.fill_(1.0)

    def set_tsi_control(self, lambda_c=0.0, track_score=0.0, intensity_score=0.0):
        self.tsi_lambda_buf.fill_(float(lambda_c))
        self.tsi_track_score_buf.fill_(float(track_score))
        self.tsi_intensity_score_buf.fill_(float(intensity_score))

        delta = float(intensity_score) - float(track_score)
        temperature = float(np.exp(-self.router_temp_kappa * abs(float(lambda_c) * delta)))
        temperature = max(self.router_temp_min, min(self.router_temp_max, temperature))
        self.router_temperature_buf.fill_(temperature)

    def get_tsi_control(self):
        return {
            'lambda_c': float(self.tsi_lambda_buf.item()),
            'track_score': float(self.tsi_track_score_buf.item()),
            'intensity_score': float(self.tsi_intensity_score_buf.item()),
            'temperature': float(self.router_temperature_buf.item()),
        }

    def _router_delta(self):
        return self.tsi_intensity_score_buf - self.tsi_track_score_buf

    def _apply_router_control(self, logits):
        """
        Apply TSI-aware bias and temperature to router logits.

        Positive delta = favor larger-scale experts (intensity side).
        Negative delta = favor smaller-scale experts (track side).
        """
        lambda_c = self.tsi_lambda_buf.to(logits.device)
        if float(lambda_c.item()) <= 0.0:
            return logits

        delta = self._router_delta().to(logits.device)
        bias = self.router_bias_strength * lambda_c * delta * self.expert_scale_prior.to(logits.device)
        logits = logits + bias.unsqueeze(0)

        temperature = self.router_temperature_buf.to(logits.device).clamp(
            min=self.router_temp_min,
            max=self.router_temp_max,
        )
        logits = logits / temperature
        return logits

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """_input: (batch, decoder_h_dim - noise_first_dim)."""
        if not self.noise_dim:
            return _input

        device = _input.device
        use_global = (self.noise_mix_type == 'global') and (seq_start_end is not None)
        if use_global:
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim

        z_decoder = user_noise if user_noise is not None else get_noise(noise_shape, self.noise_type, device)

        if use_global:
            pieces = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                z_vec = z_decoder[idx].view(1, -1).repeat(end - start, 1)
                pieces.append(torch.cat([_input[start:end], z_vec], dim=1))
            return torch.cat(pieces, dim=0)

        return torch.cat([_input, z_decoder], dim=1)

    def mlp_decoder_needed(self):
        return bool(self.noise_dim or self.pooling_type or (self.encoder_h_dim != self.decoder_h_dim))

    def get_router_logits(self, enc_h):
        logits = self.net_chooser(enc_h).reshape(-1, self.num_gs)
        logits = self._apply_router_control(logits)
        return logits

    def get_samples(self, enc_h, num_samples=6):
        """
        enc_h: (batch, encoder_h_dim)
        Returns:
        - net_chooser_out: (batch, num_gs)
        - sampled_gen_idxs: numpy (batch, num_samples)
        """
        net_chooser_out = self.get_router_logits(enc_h)
        dist = Categorical(logits=net_chooser_out)
        sampled_gen_idxs = dist.sample((num_samples,)).transpose(0, 1)
        return net_chooser_out, sampled_gen_idxs.detach().cpu().numpy()

    def mix_noise(self, final_encoder_h, seq_start_end, batch, user_noise=None):
        """final_encoder_h: (1,b,encoder_h_dim) or (b,encoder_h_dim)."""
        device = final_encoder_h.device
        mlp_in = final_encoder_h.view(-1, self.encoder_h_dim)

        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_in)
        else:
            noise_input = mlp_in

        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = decoder_h.view(-1, batch, self.decoder_h_dim)
        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim, device=device)
        return decoder_h, decoder_c

    def forward(
        self,
        obs_traj,
        obs_traj_rel,
        seq_start_end,
        image_obs,
        env_data,
        num_samples=1,
        all_g_out=False,
        predrnn_img=None,
        user_noise=None,
    ):
        """
        obs_traj_rel: (obs_len, b, 4)
        image_obs: (b, c, obs_len, h, w)
        Return:
        - pred_traj_fake_rel_nums: (pred_len, num_samples or num_gs, b, 4)
        - image_out: (b, c, obs_len+pred_len, h, w)
        - net_chooser_out: (b, num_gs)
        - sampled_gen_idxs: numpy (b, num_samples)
        """
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

        if predrnn_img is None:
            predrnn_out = self.Unet(image_obs)
        else:
            predrnn_out = predrnn_img

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

        if all_g_out:
            preds_rel = []
            with torch.no_grad():
                state_tuple = self.mix_noise(dec_h.unsqueeze(0), seq_start_end, batch, user_noise=user_noise)
                for g in self.gs:
                    pred_traj_fake_rel, _ = g(
                        obs_traj,
                        obs_traj_rel,
                        last_pos,
                        last_pos_rel,
                        state_tuple,
                        seq_start_end,
                        decoder_img,
                        last_img,
                    )
                    preds_rel.append(pred_traj_fake_rel.unsqueeze(1))

            pred_traj_fake_rel_nums = torch.cat(preds_rel, dim=1)
            net_chooser_out, sampled_gen_idxs = self.get_samples(dec_h_evn, num_samples)
            return pred_traj_fake_rel_nums, image_out, net_chooser_out, sampled_gen_idxs

        net_chooser_out, sampled_gen_idxs = self.get_samples(dec_h_evn, num_samples)

        preds_rel_samples = []
        for sample_i in range(num_samples):
            gs_index = sampled_gen_idxs[:, sample_i]
            unique_gs = np.unique(gs_index)

            batch_pieces = []
            original_indices = []
            for g_idx in unique_gs:
                now_data_index = np.where(gs_index == g_idx)[0]
                if len(now_data_index) < 1:
                    continue

                idx_t = torch.as_tensor(now_data_index, dtype=torch.long, device=device)

                sub_obs_traj = obs_traj[:, idx_t]
                sub_obs_traj_rel = obs_traj_rel[:, idx_t]
                sub_last_pos = last_pos[idx_t]
                sub_last_pos_rel = last_pos_rel[idx_t]
                sub_decoder_img = decoder_img[:, idx_t]
                sub_last_img = last_img[idx_t]

                sub_dec_h = dec_h[idx_t].unsqueeze(0)
                sub_seq_start_end = None
                if seq_start_end is not None and self.noise_mix_type == 'global':
                    # Conservative fallback: use local noise in sub-batches if group structure is unknown.
                    sub_seq_start_end = None

                state_tuple = self.mix_noise(
                    sub_dec_h,
                    sub_seq_start_end,
                    len(now_data_index),
                    user_noise=user_noise,
                )

                out_rel, _ = self.gs[g_idx](
                    sub_obs_traj,
                    sub_obs_traj_rel,
                    sub_last_pos,
                    sub_last_pos_rel,
                    state_tuple,
                    sub_seq_start_end,
                    sub_decoder_img,
                    sub_last_img,
                )

                batch_pieces.append(out_rel)
                original_indices.append(idx_t)

            all_pieces = torch.cat(batch_pieces, dim=1)
            all_indices = torch.cat(original_indices, dim=0)
            _, rev_idx = torch.sort(all_indices)
            pred_traj_fake_rel_ordered = all_pieces[:, rev_idx, :]
            preds_rel_samples.append(pred_traj_fake_rel_ordered.unsqueeze(1))

        pred_traj_fake_rel_nums = torch.cat(preds_rel_samples, dim=1)
        return pred_traj_fake_rel_nums, image_out, net_chooser_out, sampled_gen_idxs


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        num_layers=1,
        activation='relu',
        batch_norm=True,
        dropout=0.0,
        d_type='local',
    ):
        super().__init__()

        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.seq_len = self.obs_len + self.pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.img_embedding = nn.Linear(64 * 64, embedding_dim)
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

        if d_type == 'global':
            pass

    def forward(self, traj, traj_rel, seq_start_end, img):
        """
        traj_rel: (obs+pred, batch, 4)
        img: (b, c, obs+pred, h, w)
        """
        b, c, t, h, w = img.shape
        img_embed_input = img.view(b, t, -1)
        img_embed = self.img_embedding(img_embed_input).permute(1, 0, 2)

        final_h = self.encoder(traj_rel, img_embed)['final_h'][0]
        classifier_input = final_h[-1] if final_h.dim() == 3 else final_h
        classifier_input = classifier_input.reshape(-1, self.h_dim)

        scores = self.real_classifier(classifier_input)
        return scores, classifier_input
