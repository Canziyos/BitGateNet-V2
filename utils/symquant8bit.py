import torch
from .log_tools import LoggerUnit

class SymQuant8bit:
    """Symmetric fake‑quantiser with optional group‑wise scaling."""
    # Construction.
    # --------------------- #
    def __init__(
        self,
        num_bits: int = 8,
        group_dim: int = 1,
        group_size: int = 1,
        eps: float = 1e-5,
        quantscale: float = 1.0,
        enabled: bool = True,
        return_int: bool = False,
    ) -> None:
        self.num_bits = num_bits
        self.qmax = (1 << (num_bits - 1)) - 1  # 127 for 8‑bit.
        self.group_dim = group_dim
        self.group_size = group_size
        self.eps = eps
        self.quantscale = quantscale
        self.enabled = enabled
        self.return_int = return_int
        self.logger = LoggerUnit("SymQuant8bit").get_logger()

    # ------------------------------------------------------------------ #
    # Helpers.
    # ------------------------------------------------------------------ #
    def _post_process(self, q_x: torch.Tensor):
        if self.return_int:
            return q_x.to(torch.int8)
        return q_x

    def _scalar_quant(self, x: torch.Tensor):
        max_val = x.abs().clamp(min=self.eps)
        scale = (self.qmax * self.quantscale) / max_val
        q_x = (x * scale).round().clamp(-self.qmax, self.qmax)
        return self._post_process(q_x), scale

    # ------------------------------------------------------------------ #
    # Public API.
    # ------------------------------------------------------------------ #
    def quantize(self, x: torch.Tensor, verbose: bool = False):
        """Return fake‑quantised tensor and its scale."""
        if not self.enabled:
            return x, torch.tensor(1.0, device=x.device)

        # 0D / 1D fall back to scalar path.
        if x.dim() <= 1:
            q_x, scale = self._scalar_quant(x)
            if verbose:
                self.logger.debug(f"D{ x.dim() }: scale={ scale }")
            return q_x, scale

        # ---------------- N‑D group‑wise path ---------------- #
        c = x.size(self.group_dim)
        g = (c + self.group_size - 1) // self.group_size  # ceil division.

        # Bring group_dim to front then flatten.
        x_abs = x.abs()
        x_perm = x_abs.transpose(0, self.group_dim).contiguous()
        x_group = x_perm.view(g, -1)
        max_vals = x_group.amax(dim=1).clamp(min=self.eps)  # [g]

        scale_g = (self.qmax * self.quantscale) / max_vals  # [g]

        # Expand group‑scale back to per‑channel.
        scale_c = scale_g.repeat_interleave(self.group_size)[:c]  # [c]
        scale_shape = [1] * x.dim()
        scale_shape[self.group_dim] = c
        scale_c = scale_c.view(scale_shape)

        q_x = (x * scale_c).round().clamp(-self.qmax, self.qmax)
        q_x = self._post_process(q_x)

        if verbose:
            self.logger.debug(
                f"ND: scale shape={ scale_c.shape }, qmin={ q_x.min() }, qmax={ q_x.max() }"
            )
        return q_x, scale_c

    def dequantize(self, q_x: torch.Tensor, scale: torch.Tensor):
        return q_x.float() / scale

    def apply_fake_quant(self, q_x: torch.Tensor, scale: torch.Tensor, x: torch.Tensor):
        if not self.enabled:
            return x
        return x + (q_x.float() / scale - x).detach()

    def export_int8(self, x: torch.Tensor):
        """Return (int8_weights, scale) ready for C inference."""
        q_x, scale = self.quantize(x)
        return q_x.to(torch.int8), scale

    def __str__(self) -> str:
        return (
            f"SymQuant{ self.num_bits }bit(group_dim={ self.group_dim }, "
            f"group_size={ self.group_size }, enabled={ self.enabled })"
        )
