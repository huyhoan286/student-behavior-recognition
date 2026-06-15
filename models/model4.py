import torch.nn as nn
from torchvision.models.video import Swin3D_T_Weights, swin3d_t

from config import M4, NUM_CLASSES


class Model4(nn.Module):
    """
    Video Swin Transformer Tiny, pretrained trên Kinetics-400.
    Input: [B, T, C, H, W] với T=16 frames.
    Hierarchical shifted-window attention học spatial-temporal features cùng lúc.
    """

    def __init__(self):
        super().__init__()
        self.swin = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)

        # Thay head 400-class bằng head mới cho NUM_CLASSES
        in_features = self.swin.head.in_features   # 768
        self.swin.head = nn.Sequential(
            nn.Dropout(M4["dropout"]),
            nn.Linear(in_features, NUM_CLASSES),
        )

    def get_encoder_params(self):
        """Params của encoder (không gồm head) — dùng cho differential LR."""
        head_ids = {id(p) for p in self.swin.head.parameters()}
        return [p for p in self.swin.parameters() if id(p) not in head_ids]

    def get_head_params(self):
        return list(self.swin.head.parameters())

    def forward(self, x):
        # x: [B, T, C, H, W] → Swin cần [B, C, T, H, W]
        return self.swin(x.permute(0, 2, 1, 3, 4).contiguous())
