# conv2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import SymQuant8bit


class Conv2d(nn.Conv2d):
    """2-D convolution with symmetric fake-quant support (per-output-channel).
    Caches de-quantised float weights/bias at eval() for speed.
    All sentences end with a period.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        quantizer: SymQuant8bit | None = None,
        out_int: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        # *** Change: per-output-channel (group_dim=0) default. ***
        self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)
        self.out_int = out_int
        self.register_buffer("_w_fq", None)  # cached de-quantised weight
        self.register_buffer("_b_fq", None)  # cached de-quantised bias

    # ------------------------------------------------------------------ #
    # Helpers.                                                           #
    # ------------------------------------------------------------------ #
    def _cache_eval_tensors(self) -> None:
        """Quantise weights/bias once and cache float copies."""
        w_q, w_scale = self.quantizer.quantize(self.weight)
        if self.out_int:
            w_fq = w_q.to(torch.int8)
            w_fq = w_fq.float() / w_scale
        else:
            w_fq = self.quantizer.apply_fake_quant(w_q, w_scale, self.weight)
        self._w_fq = w_fq

        if self.bias is not None:
            b_q, b_scale = self.quantizer.quantize(self.bias)
            if self.out_int:
                b_fq = b_q.to(torch.int8)
                b_fq = b_fq.float() / b_scale
            else:
                b_fq = self.quantizer.apply_fake_quant(b_q, b_scale, self.bias)
            self._b_fq = b_fq

    # ------------------------------------------------------------------ #
    # Forward.                                                           #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # Float path when quantiser disabled.
        if not self.quantizer.enabled:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        # Training: re-quant every call to keep gradients live.
        if self.training:
            x_q, x_scale = self.quantizer.quantize(x)
            x_fq = self.quantizer.apply_fake_quant(x_q, x_scale, x)

            w_q, w_scale = self.quantizer.quantize(self.weight)
            w_fq = self.quantizer.apply_fake_quant(w_q, w_scale, self.weight)

            if self.bias is not None:
                b_q, b_scale = self.quantizer.quantize(self.bias)
                b_fq = self.quantizer.apply_fake_quant(b_q, b_scale, self.bias)
            else:
                b_fq = None

            return F.conv2d(
                x_fq,
                w_fq,
                b_fq,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        # Eval: cache once, then reuse.
        if self._w_fq is None:
            self._cache_eval_tensors()

        x_q, x_scale = self.quantizer.quantize(x)
        x_fq = self.quantizer.apply_fake_quant(x_q, x_scale, x)
        return F.conv2d(
            x_fq,
            self._w_fq,
            self._b_fq if self.bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

