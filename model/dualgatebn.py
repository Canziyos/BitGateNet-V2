import torch.nn as nn

from .gateresidual import GateResidual
from .batchnorm2d import BatchNorm2D
from utils import SymQuant8bit


class DualGateBN(nn.Module):
    """Two GateResidual + BatchNorm2D blocks with a residual skip.
    Structure: (Gate => BN) × 2, followed by skip‑connection add.
    """

    def __init__(
        self,
        in_channels: int,
        mid0: int,
        out0: int,
        mid1: int,
        out1: int,
        *,
        quantizer: SymQuant8bit | None = None,
    ) -> None:
        super().__init__()
        q = quantizer or SymQuant8bit(quantscale=0.25)

        # First Gate + BN.
        self.block1 = nn.Sequential(
            GateResidual(in_channels, mid0, out0, quantizer=q),
            BatchNorm2D(in_channels, quantizer=q),
            # nn.Identity()

        )

        # Second Gate + BN.
        self.block2 = nn.Sequential(
            GateResidual(in_channels, mid1, out1, quantizer=q),
            BatchNorm2D(in_channels, quantizer=q),
            # nn.Identity() 
        )

    # Forward.
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x
