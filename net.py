import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.h_1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 只出 logits
        self.action_dim = action_dim

    def forward(self, x):
        # 允许 1D/2D 输入
        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        h = self.h_1(self.fc1(x))
        logits = self.fc2(h)                 # 不做 softmax，不做掩码
        if logits.size(0) == 1:
            logits = logits.squeeze(0)
        return logits

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, global_emb_dim=0):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim+global_emb_dim, hidden_dim)
        self.h_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, global_emb=None):
        x = torch.cat([x, global_emb], dim=1) if global_emb is not None  else x
        x = F.relu(self.h_1(F.relu(self.fc1(x))))
        return self.fc2(x)


class VAE(nn.Module):
    """
    卷积版 VAE：
    - 输入: [B, C=state_dim, H, W]
    - 编码: Conv→Conv→Conv(stride=2) 得到 [B, 128, H_out, W_out]
    - 展平: F_enc = 128 * H_out * W_out  （由真实卷积输出来决定）
    - 全连接: fc_mu / fc_logvar / fc_decode 在第一次前向时按 F_enc 自动初始化
    - 解码: fc_decode → [B,128,H_out,W_out] → ConvTranspose → [B, C, H, W]
    """
    def __init__(self, state_dim: int, latent_dim: int):
        super().__init__()
        assert state_dim > 0 and latent_dim > 0
        self.state_dim  = int(state_dim)
        self.latent_dim = int(latent_dim)

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(self.state_dim, 128, kernel_size=3, stride=1, padding=1),  # [B, 32, H,   W  ]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),              # [B, 64, H,   W  ]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),             # [B,128, H/2, W/2] (向下取整)
            nn.ReLU()
        )

        # 延迟初始化：看到真实 conv 输出后再建全连接层
        self.fc_mu     = None  # nn.Linear(F_enc, latent_dim)
        self.fc_logvar = None  # nn.Linear(F_enc, latent_dim)
        self.fc_decode = None  # nn.Linear(latent_dim, F_enc)

        # --- Decoder ---
        # 第一层 stride=2 的 ConvTranspose2d 在 forward 前会动态设置 output_padding=(op_h, op_w)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256,  kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256,  128,  kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,  self.state_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # 记录卷积输出的空间尺寸，供解码 reshape 使用
        self._enc_shape = None   # (C_enc=128, H_out, W_out)
        self._F_enc     = None   # 128 * H_out * W_out
        self._in_hw     = None   # (H_in, W_in)，在 encode() 里记录

    # ---- 编码：卷积 → 展平 → (lazy) 全连接 ----
    def encode(self, x: torch.Tensor):
        # x: [B, C, H, W]  —— 记录输入的原始空间尺寸，供解码对齐
        self._in_hw = (int(x.shape[-2]), int(x.shape[-1]))  # (H, W)
        h = self.encoder(x)  # [B, 128, H_out, W_out]
        B, Cc, Ho, Wo = h.shape
        F_enc = Cc * Ho * Wo
        if (self._enc_shape is None) or (self._F_enc != F_enc):
            # 第一次或网格尺寸变化时，按真实 F_enc 初始化/重建全连接
            device = x.device
            self._enc_shape = (Cc, Ho, Wo)
            self._F_enc     = F_enc
            self.fc_mu     = nn.Linear(F_enc, self.latent_dim).to(device)
            self.fc_logvar = nn.Linear(F_enc, self.latent_dim).to(device)
            self.fc_decode = nn.Linear(self.latent_dim, F_enc).to(device)

        z = h.reshape(B, F_enc)                   # [B, F_enc]
        mu     = self.fc_mu(z)                    # [B, L]
        logvar = self.fc_logvar(z)                # [B, L]
        return mu, logvar

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _calc_output_padding_hw(H_in: int, W_in: int, Ho: int, Wo: int,
                                k: int = 3, s: int = 2, p: int = 1):
        """
        计算第一层 ConvTranspose2d 在逐轴上的 output_padding：(op_h, op_w)
        反卷积无 output_padding 时输出：
            H0 = (Ho - 1) * s - 2p + k
            W0 = (Wo - 1) * s - 2p + k
        希望达到 H_in, W_in，则：
            op_h = clamp(H_in - H0, 0, s-1)
            op_w = clamp(W_in - W0, 0, s-1)
        """
        H0 = (Ho - 1) * s - 2 * p + k
        W0 = (Wo - 1) * s - 2 * p + k
        op_h = max(0, min(s - 1, int(H_in - H0)))
        op_w = max(0, min(s - 1, int(W_in - W0)))
        return (op_h, op_w)

    def decode(self, z: torch.Tensor):
        # z: [B, L]
        assert self._enc_shape is not None and self._F_enc is not None, "VAE.decode 在 encode 之前被调用。"
        B = z.size(0)
        h = self.fc_decode(z)                     # [B, F_enc]
        Cc, Ho, Wo = self._enc_shape
        h = h.view(B, Cc, Ho, Wo)                 # [B, 128, H_out, W_out]

        # === 动态设置第一层反卷积的 output_padding（逐轴 tuple） ===
        # encode() 已记录原始输入 H_in, W_in；第一层 ConvTranspose2d 的超参为 k=3, s=2, p=1
        H_in, W_in = getattr(self, "_in_hw", (Ho * 2, Wo * 2))
        op_h, op_w = self._calc_output_padding_hw(H_in, W_in, Ho, Wo, k=3, s=2, p=1)

        # 修改 Sequential 第 0 层（ConvTranspose2d）的 output_padding
        deconv1 = self.decoder[0]
        # 确保不越界（PyTorch 要求 0 <= op < stride）
        op_h = min(max(op_h, 0), deconv1.stride[0] - 1)
        op_w = min(max(op_w, 0), deconv1.stride[1] - 1)
        deconv1.output_padding = (op_h, op_w)

        # 反卷积解码
        x_recon = self.decoder(h)                 # [B, C, H', W']

        # 兜底对齐（通常不会触发；若后续你改了 kernel/stride/padding，可避免1像素误差）
        if (x_recon.shape[-2] != H_in) or (x_recon.shape[-1] != W_in):
            x_recon = F.interpolate(x_recon, size=(H_in, W_in), mode="nearest")

        return x_recon

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def representation(self, x: torch.Tensor):
        mu, _ = self.encode(x)
        return mu

    



