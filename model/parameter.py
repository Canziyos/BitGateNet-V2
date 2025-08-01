import torch
import torch.nn as nn
from utils import SymQuant8bit


class Parameter(nn.Module):
    """Learnable scalar / tensor with symmetric fake‑quant support.
    Caches de‑quantised float tensor in eval() for speed.
    """
    def __init__(
        self,
        val: torch.Tensor,
        *,
        quantizer: SymQuant8bit | None = None,
        return_int: bool = False,
    ) -> None:
        super().__init__()
        self.param = nn.Parameter(val)
        self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)
        self.return_int = return_int
        self.register_buffer("_p_cache", None)  # float cache for eval().

    # Helpers.
    # ------- #
    def _fq_param(self) -> torch.Tensor:
        q_p, s_p = self.quantizer.quantize(self.param)
        return self.quantizer.apply_fake_quant(q_p, s_p, self.param)

    def _prepare_eval_cache(self):
        if self._p_cache is not None:
            return
        with torch.no_grad():
            q_p, s_p = self.quantizer.quantize(self.param)
            if self.return_int:
                # De‑quantise once so downstream ops use float and avoid dtype clash.
                self._p_cache = self.quantizer.dequantize(q_p, s_p)
            else:
                self._p_cache = self.quantizer.apply_fake_quant(q_p, s_p, self.param)

    # Forward.
    # -------- #
    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        if not self.quantizer.enabled:
            p = self.param
        elif self.training:
            p = self._fq_param()
        else:  # eval fast path.
            self._prepare_eval_cache()
            p = self._p_cache

        return p if x is None else x * p
