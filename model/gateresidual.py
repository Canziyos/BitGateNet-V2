import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2d import Conv2d
from .depthwise import DepthwiseSeparable
from .parameter import Parameter
from utils import SymQuant8bit, LoggerUnit


class GateResidual(nn.Module):
    """Gated residual block.
    Down‑sample → bottleneck → up‑sample path produces a sigmoid mask that modulates the residual.
    Set `depthwise=True` to replace pointwise 1×1 convs with depth‑wise separable blocks.
    All sentences end with a period.
    """

    def __init__(
        self,
        in_channels: int,
        mid_ch: int,
        out_ch: int,
        *,
        quantizer: SymQuant8bit | None = None,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.q = quantizer or SymQuant8bit(quantscale=1.0)
        self.log = LoggerUnit("GateResidual").get_logger()

        Conv = DepthwiseSeparable if depthwise else Conv2d

        # Condense spatial.
        self.condense = nn.MaxPool2d(2)
        # Bottleneck conv chain.
        self.group_conv = Conv(in_channels, mid_ch, kernel_size=1, stride=1, padding=0, quantizer=self.q)
        self.point_conv = Conv(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, quantizer=self.q)
        # Expand channels back.
        self.expand_conv = Conv(out_ch, in_channels, kernel_size=1, stride=1, padding=0, quantizer=self.q)
        # Restore spatial dims.
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # Learnable global scale (scalar).
        self.scale = Parameter(torch.tensor(1.0), quantizer=self.q)

    # ------------------------------------------------------------------ #
    # Forward.                                                           #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        residual = x

        k = self.condense(x)
        k = F.relu(self.group_conv(k))
        k = F.relu(self.point_conv(k))
        a = F.interpolate(k, size=residual.shape[-2:], mode="nearest")
        a = self.expand_conv(a)

        s = torch.sigmoid(a)
        if self.q.enabled and self.training:
            s_q, s_scale = self.q.quantize(s)
            s = self.q.apply_fake_quant(s_q, s_scale, s)

        v = residual * s * self.scale().view(1, 1, 1, 1)
        return v + residual
