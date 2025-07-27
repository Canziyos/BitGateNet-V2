import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv2d import Conv2d
from .depthwise import DepthwiseSeparable
from .parameter import Parameter
from utils import SymQuant8bit, LoggerUnit


class GateResidual(nn.Module):
    """
    Gated residual block with optional depthwise separable convolutions.
    Workflow:
      1) Downsample input (spatial condense).
      2) Bottleneck 1×1 convolutions (or depthwise separable if enabled).
      3) Upsample back to match residual dimensions.
      4) Produce gate via tanh (range [-1, 1]) or alternative functions.
      5) Apply quantization-aware fake-quant to gate output if QAT is active.
      6) Combine with residual using weighted residual add.

    Args:
        in_channels: Input channels.
        mid_ch: Bottleneck (hidden) channels.
        out_ch: Output channels (before expanding back to in_channels).
        quantizer: Shared SymQuant8bit instance for QAT (default enabled).
        depthwise: If True, replaces 1×1 Conv with DepthwiseSeparable variant.
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

        # Use depthwise separable or standard Conv2d depending on flag.
        Conv = DepthwiseSeparable if depthwise else Conv2d

        # Downsample spatial dimensions by factor of 2.
        self.condense = nn.MaxPool2d(2)

        # Bottleneck path: in -> mid -> out -> expand to in_channels.
        self.group_conv = Conv(in_channels, mid_ch, kernel_size=1, stride=1, padding=0, quantizer=self.q)
        self.point_conv = Conv(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, quantizer=self.q)
        self.expand_conv = Conv(out_ch, in_channels, kernel_size=1, stride=1, padding=0, quantizer=self.q)

        # Upsample back to original spatial resolution.
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Learnable global scale parameter (scalar).
        self.scale = Parameter(torch.tensor(1.0), quantizer=self.q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Downsample -> bottleneck -> upsample
        k = self.condense(x)
        k = F.relu(self.group_conv(k))
        k = F.relu(self.point_conv(k))
        a = F.interpolate(k, size=residual.shape[-2:], mode="nearest")
        a = self.expand_conv(a)

        # -----------------------------
        # Gate function (choose one)
        # -----------------------------
        s = torch.tanh(a)  # Default: Tanh gate

        # -----------------------------
        # Quantization-aware fake quant
        # -----------------------------
        if self.q.enabled and self.training:
            s_q, s_scale = self.q.quantize(s)
            s = self.q.apply_fake_quant(s_q, s_scale, s)

        # Debug logging
        # if self.training:
        #     print(
        #         f"[GateResidual] Mask stats -> min:{s.min().item():.3f} "
        #         f"max:{s.max().item():.3f} mean:{s.mean().item():.3f}"
        #     )
        #     print(f"[GateResidual] Scale param -> {self.scale().item():.3f}")

        # -----------------------------
        # Weighted residual add
        # -----------------------------
        return residual + (residual * s * self.scale().view(1, 1, 1, 1))
