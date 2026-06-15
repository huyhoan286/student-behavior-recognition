import torch.nn as nn
from transformers import TimesformerForVideoClassification

from config import M3, NUM_CLASSES


class Model3(nn.Module):
    """
    TimeSformer-base fine-tuned từ Kinetics-400.
    Input: [B, T, C, H, W] với T=8 frames.
    Divided space-time attention: học spatial và temporal attention riêng biệt.
    """

    def __init__(self):
        super().__init__()
        self.timesformer = TimesformerForVideoClassification.from_pretrained(
            M3["model_name"],
            num_labels       = NUM_CLASSES,
            ignore_mismatched_sizes = True,   # classifier head được reset
        )
        # expose encoder cho trainer (freeze_epochs=0 nên không dùng nhưng cần có)
        self.backbone = self.timesformer.timesformer

    def forward(self, x):
        # x: [B, T, C, H, W]
        return self.timesformer(pixel_values=x).logits
