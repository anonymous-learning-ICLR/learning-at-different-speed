import torch
from torch import nn
import torch.nn.functional as F

'''
The shape of Conv3d input is (batch size, channel, sequence, height, width)
'''


class Conv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(True),
        )

        self.residual = nn.Conv3d(
            in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.residual(x)


class Down(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
            Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, x1_in, x2_in, out_channel, kernel_size, stride, padding):
        super(Up, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose3d(
                x1_in,
                x1_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(),
        )

        self.conv = Conv3d(x1_in + x2_in, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # x shape: [N, C, D, H, W]
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)

        # F.pad order: (W_left, W_right, H_left, H_right, D_left, D_right)
        x1 = F.pad(
            x1,
            [
                diffW // 2, diffW - diffW // 2,
                diffH // 2, diffH - diffH // 2,
                diffD // 2, diffD - diffD // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel_list, out_channel, kernel_size, stride, padding):
        super(OutConv, self).__init__()

        self.up_list = []
        for i, channel in enumerate(in_channel_list):
            if i == len(in_channel_list) - 1:
                continue
            scale = 2 ** ((len(in_channel_list) - 1) - i)
            self.up_list.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        channel,
                        channel,
                        kernel_size=[1, scale, scale],
                        stride=[1, scale, scale],
                        padding=padding,
                        bias=True,
                    ),
                    nn.ReLU(),
                    nn.Conv3d(channel, in_channel_list[-1], kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm3d(in_channel_list[-1]),
                    nn.ReLU(inplace=True),
                )
            )
        self.up_list = nn.ModuleList(self.up_list)

        self.conv = nn.Sequential(
            nn.Conv3d(in_channel_list[-1], out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x6, x7, x8, x9 = tuple(x)
        x6 = self.up_list[0](x6)
        x7 = self.up_list[1](x7)
        x8 = self.up_list[2](x8)

        # All tensors now have the same channel dimension; concatenate along time/depth.
        x_last = torch.cat([x6, x7, x8, x9], dim=2)
        return self.conv(x_last)


class Unet3D(nn.Module):
    """
    Dynamic 3D U-Net for variable obs_len / pred_len.

    Input:
        x: [B, C, obs_len, H, W]

    Output:
        out: [B, out_channel, obs_len + pred_len - 1, H, W]

    Notes:
    - The encoder/decoder temporal structure keeps the original design.
    - The final temporal kernel is computed dynamically so that the output
      time dimension becomes exactly (obs_len + pred_len - 1).
    """
    def __init__(self, in_channel, out_channel, obs_len=8, pred_len=4):
        super(Unet3D, self).__init__()

        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)

        if self.obs_len < 4:
            raise ValueError(f"obs_len must be >= 4, got {self.obs_len}")
        if self.pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {self.pred_len}")

        self.inc = Conv3d(in_channel, 16, kernel_size=3, stride=1, padding=1)
        self.down1 = Down(16, 32, kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.down2 = Down(32, 64, kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.down3 = Down(64, 128, kernel_size=[2, 2, 2], stride=[2, 2, 2])
        self.down4 = Down(128, 128, kernel_size=[2, 2, 2], stride=[2, 2, 2])

        self.up1 = Up(128, 128, 64, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.up2 = Up(64, 64, 32, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.up3 = Up(32, 32, 16, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)
        self.up4 = Up(16, 16, 16, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)

        # Time-depth after concatenating x6, x7, x8, x9 in OutConv:
        # x6 has floor(obs_len / 2), and x7/x8/x9 each have obs_len.
        t_sum = (self.obs_len // 2) + 3 * self.obs_len
        out_t = self.obs_len + self.pred_len - 1
        k_t = t_sum - out_t + 1

        if k_t < 1:
            raise ValueError(
                f"Invalid temporal kernel size: k_t={k_t}. "
                f"Got obs_len={self.obs_len}, pred_len={self.pred_len}. "
                f"Current architecture supports settings where "
                f"obs_len//2 + 3*obs_len >= obs_len + pred_len."
            )

        print(
            f"Unet3D initialized with obs_len={self.obs_len}, "
            f"pred_len={self.pred_len}, t_sum={t_sum}, out_t={out_t}, kernel_t={k_t}"
        )

        self.outc = OutConv(
            [64, 32, 16, 16],
            out_channel,
            kernel_size=[k_t, 1, 1],
            stride=[1, 1, 1],
            padding=0,
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        out = self.outc([x6, x7, x8, x9])
        return out


