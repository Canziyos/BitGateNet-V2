"""BitGateNet‑V2.
keyword‑spotting CNN with gate residuals and depth‑wise head.
"""

import torch.nn as nn
import torch.nn.functional as F

from utils import SymQuant8bit, LoggerUnit
from .conv2d import Conv2d
from .linear import Linear
from .dualgatebn import DualGateBN
from .depthwise import DepthwiseSeparable
from .gateresidual import GateResidual


class BitGateNetV2(nn.Module):
    """Backbone network: Conv => (GateBN ×2) => DW head => FC."""

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

        c1 = _c(16)
        c_mid = _c(32)
        c2 = _c(32)

        # Stem.
        self.conv1 = Conv2d(1, c1, kernel_size=3, padding=1, quantizer=q)

        # debug: replace with torch conv2d
        # Stem (plain PyTorch Conv2d, no quantization).
        #self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)

        # Gate residual stack.
        self.block1 = DualGateBN(c1, c_mid, c1, c_mid, c1, quantizer=q)
        self.block2 = DualGateBN(c1, c_mid, c1, c_mid, c1, quantizer=q)

        # debug1:
        # self.block1 = GateResidual(c1, c_mid, c1, quantizer=q)
        # self.block2 = GateResidual(c1, c_mid, c1, quantizer=q)
        #self.block2 = nn.Identity()

        # debug2: Conv1 => DepthwiseSeparable => FC
        # self.block1 = nn.Identity()
        # self.block2 = nn.Identity()

        # Depth‑wise separable stage.
        self.dw_stage = DepthwiseSeparable(c1, c2, quantizer=q)

        #debug: Conv1 => Global Pool => FC.
        #self.dw_stage = nn.Identity()

        # Head.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embed = Linear(c2, emb_dim, quantizer=q) if emb_dim else nn.Identity()
        #self.embed = Linear(c1, emb_dim, quantizer=q) if emb_dim else nn.Identity()

        fc_in = emb_dim or (c2*63)
        self.fc = Linear(fc_in, num_classes, quantizer=q)
        # Debug: If dw_stage is Identity, use c1 instead of c2
        #fc_in = emb_dim or (c1 if isinstance(self.dw_stage, nn.Identity) else c2)
        # If skipping global pool, input is c1 * 40 * 63
        #fc_in = emb_dim or (c1*63 if isinstance(self.dw_stage, nn.Identity) else c2)
        
        # fc_in = emb_dim or (c1*63)
        # self.fc = Linear(fc_in, num_classes, quantizer=q)

        self.dropout = nn.Dropout(0.2) if use_dropout else nn.Identity()
        self.fc = Linear(fc_in, num_classes, quantizer=q)

    # Forward.
    # ------------------- #
    def forward(self, x, return_features=False):
        # Initial conv + gate blocks
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.dw_stage(x)

        # ----- Feature tap (before pooling) -----
        features = x  # B*C*F (frequency/time)

        # Average over frequency (dim=2), keep time resolution
        x = x.mean(dim=2)        # now B*C*63
        x = x.flatten(1)         # flatten to B*(C*63)
        x = self.embed(x)
        x = self.dropout(x)
        logits = self.fc(x)

        # Return logits + features if requested (for feature KD)
        if return_features:
            return logits, features
        return logits


    def __str__(self):
        return "BitGateNetV2"