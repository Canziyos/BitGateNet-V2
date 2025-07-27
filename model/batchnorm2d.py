import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import SymQuant8bit


class BatchNorm2D(nn.BatchNorm2d):
    """BatchNorm2d with symmetric fake‑quant support.
    Caches de‑quantised gamma/beta in eval() to avoid per‑call quant ops.
    """
    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        quantizer: SymQuant8bit | None = None,
    ) -> None:
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.quantizer = quantizer or SymQuant8bit(group_dim=0, group_size=1)
        self.register_buffer("_w_fq", None)  # cached gamma (float)
        self.register_buffer("_b_fq", None)  # cached beta (float)

    # ---------------------------------------------------------- #
    # Helpers.
    # ---------------------------------------------------------- #
    def _fake_quant_param(self, p: torch.Tensor) -> torch.Tensor:
        p_q, p_scale = self.quantizer.quantize(p)
        return self.quantizer.apply_fake_quant(p_q, p_scale, p)

    def _cache_params(self) -> None:
        """Quantise and de‑quantise weight/bias once in eval()."""
        if self.affine:
            self._w_fq = self._fake_quant_param(self.weight)
            self._b_fq = self._fake_quant_param(self.bias)
        else:
            self._w_fq = None
            self._b_fq = None

    # ----------------------------------#
    # Forward.
    # ----------------------------------#
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantizer.enabled:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )

        if self.training:
            x_fq = self._fake_quant_param(x)
            w_fq = self._fake_quant_param(self.weight) if self.affine else None
            b_fq = self._fake_quant_param(self.bias) if self.affine else None
            return F.batch_norm(
                x_fq,
                self.running_mean,
                self.running_var,
                w_fq,
                b_fq,
                True,
                self.momentum,
                self.eps,
            )

        # Eval path.
        if self._w_fq is None and self.affine:
            self._cache_params()
        x_fq = self._fake_quant_param(x)
        return F.batch_norm(
            x_fq,
            self.running_mean,
            self.running_var,
            self._w_fq,
            self._b_fq,
            False,
            self.momentum,
            self.eps,
        )
