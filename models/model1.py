import timm
import torch.nn as nn

from config import M1, NUM_CLASSES


class Model1(nn.Module):
    """EfficientNet-B0 fine-tuned cho frame-level classification."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            M1["backbone"], pretrained=True, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(M1["dropout"]),
            nn.Linear(feat_dim, NUM_CLASSES),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)
