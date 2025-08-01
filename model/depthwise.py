import torch.nn as nn
import torch.nn.functional as F

from utils import SymQuant8bit
from .conv2d import Conv2d
from .batchnorm2d import BatchNorm2D


class DepthwiseSeparable(nn.Module):
    """Depth‑wise separable conv block: DW‑3×3 => PW‑1×1 => BN => ReLU.
    Parameters allow stride/padding tweaks so the layer can down‑sample.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        quantizer: SymQuant8bit | None = None,
    ) -> None:
        super().__init__()
        q = quantizer or SymQuant8bit()
        if padding is None:
            padding = kernel_size // 2

        self.dw = Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            quantizer=q,
        )
        self.pw = Conv2d(in_ch, out_ch, kernel_size=1, quantizer=q)
        self.bn = BatchNorm2D(out_ch, quantizer=q)
        # self.bn = nn.Identity()


    # Forward. 
    # --------------------------- #
    def forward(self, x):
        x = F.relu(self.dw(x))
        x = self.bn(self.pw(x))
        return F.relu(x)
