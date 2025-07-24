"""BitGateNet‑V2.
Compact keyword‑spotting CNN with gate residuals and depth‑wise head.
All sentences end with a period.
"""

import torch.nn as nn
import torch.nn.functional as F

from utils import SymQuant8bit, LoggerUnit
from .conv2d import Conv2d
from .linear import Linear
from .dualgatebn import DualGateBN
from .depthwise import DepthwiseSeparable


class BitGateNetV2(nn.Module):
    """Backbone network: Conv → (GateBN ×2) → DW head → FC."""

    def __init__(
        self,
        num_classes: int,
        *,
        width_mult: float = 1.0,
        quantscale: float = 1.0,
        q_en: bool = True,
        use_dropout: bool = True,
        emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.log = LoggerUnit("BitGateNetV2").get_logger()

        q = SymQuant8bit(quantscale=quantscale, enabled=q_en)

        # Helper to round widths to multiples of 8, clipped to minimum 8.
        def _c(ch: int) -> int:
            return max(8, int(round(ch * width_mult / 8)) * 8)

        c1 = _c(8)
        c_mid = _c(16)
        c2 = _c(32)

        # Stem.
        self.conv1 = Conv2d(1, c1, kernel_size=3, padding=1, quantizer=q)

        # Gate residual stack.
        self.block1 = DualGateBN(c1, c_mid, c1, c_mid, c1, quantizer=q)
        self.block2 = DualGateBN(c1, c_mid, c1, c_mid, c1, quantizer=q)

        # Depth‑wise separable stage.
        self.dw_stage = DepthwiseSeparable(c1, c2, quantizer=q)

        # Head.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embed = Linear(c2, emb_dim, quantizer=q) if emb_dim else nn.Identity()
        fc_in = emb_dim or c2
        self.dropout = nn.Dropout(0.2) if use_dropout else nn.Identity()
        self.fc = Linear(fc_in, num_classes, quantizer=q)

    # ------------------------------------------------------------------ #
    # Forward.
    # ------------------------------------------------------------------ #
    def forward(self, x):  # noqa: D401
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.dw_stage(x)
        x = self.global_pool(x).flatten(1)
        x = self.embed(x)
        x = self.dropout(x)
        return self.fc(x)

    def __str__(self):  # noqa: D401
        return "BitGateNetV2"