import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import SymQuant8bit


class Linear(nn.Linear):
    """Fully‑connected layer with symmetric fake‑quant support.
    Caches de‑quantised float weights and bias once in eval() for speed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        quantizer: SymQuant8bit | None = None,
        out_int: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)
        self.out_int = out_int
        # Cached tensors for eval(.
        self.register_buffer("_w_fq", None)
        self.register_buffer("_b_fq", None)


    # Helpers.
    def _cache_eval_tensors(self) -> None:
        """Quantise once and cache float tensors for eval()."""
        w_q, w_scale = self.quantizer.quantize(self.weight)
        self._w_fq = self.quantizer.apply_fake_quant(w_q, w_scale, self.weight)
        if self.bias is not None:
            b_q, b_scale = self.quantizer.quantize(self.bias)
            self._b_fq = self.quantizer.apply_fake_quant(b_q, b_scale, self.bias)
        else:
            self._b_fq = None


    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Float path when quantiser disabled.
        if not self.quantizer.enabled:
            return F.linear(x, self.weight, self.bias)

        # Training: re‑quant every call for gradient flow.
        if self.training:
            x_q, x_scale = self.quantizer.quantize(x)
            x_fq = self.quantizer.apply_fake_quant(x_q, x_scale, x)

            w_q, w_scale = self.quantizer.quantize(self.weight)
            w_fq = self.quantizer.apply_fake_quant(w_q, w_scale, self.weight)

            if self.bias is None:
                return F.linear(x_fq, w_fq, None)
            b_q, b_scale = self.quantizer.quantize(self.bias)
            b_fq = self.quantizer.apply_fake_quant(b_q, b_scale, self.bias)
            return F.linear(x_fq, w_fq, b_fq)

        # Eval: cache once, then reuse.
        if self._w_fq is None:
            self._cache_eval_tensors()

        x_q, x_scale = self.quantizer.quantize(x)
        x_fq = self.quantizer.apply_fake_quant(x_q, x_scale, x)
        return F.linear(x_fq, self._w_fq, self._b_fq)
