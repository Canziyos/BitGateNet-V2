import unittest
import torch

# Adjust these two import paths if the project layout differs.
from model import DepthwiseSeparable
from utils import SymQuant8bit


class TestDepthwiseSeparable(unittest.TestCase):
    """Unit-tests for the depth-wise separable quantised conv block."""

    def _make_block(self, in_ch=16, out_ch=32, stride=1, q_en=True):
        quant = SymQuant8bit(enabled=q_en)
        return DepthwiseSeparable(in_ch, out_ch, stride=stride, quantizer=quant)

    # ------------------------------------------------------------------ #
    # Forward path: shapes & dtypes.                                      #
    # ------------------------------------------------------------------ #
    def test_forward_shape_stride1(self):
        block = self._make_block(in_ch=16, out_ch=32, stride=1)
        x = torch.randn(4, 16, 32, 32)
        y = block(x)
        self.assertEqual(y.shape, (4, 32, 32, 32), "Shape mismatch at stride 1.")

    def test_forward_shape_stride2(self):
        block = self._make_block(in_ch=16, out_ch=32, stride=2)
        x = torch.randn(2, 16, 32, 32)
        y = block(x)
        self.assertEqual(y.shape, (2, 32, 16, 16), "Shape mismatch at stride 2.")

    # ------------------------------------------------------------------ #
    # Quantiser disabled path.                                           #
    # ------------------------------------------------------------------ #
    def test_forward_without_quant(self):
        block = self._make_block(in_ch=8, out_ch=16, stride=1, q_en=False)
        x = torch.randn(1, 8, 16, 16)
        y = block(x)
        self.assertEqual(y.shape, (1, 16, 16, 16), "Float path shape bad.")

    # ------------------------------------------------------------------ #
    # Gradient propagation.                                              #
    # ------------------------------------------------------------------ #
    def test_gradients(self):
        block = self._make_block()
        block.train()
        x = torch.randn(3, 16, 20, 20, requires_grad=True)
        y = block(x).sum()
        y.backward()
        total_grad = sum(p.grad.abs().sum() for p in block.parameters())
        self.assertGreater(total_grad.item(), 0, "No gradients flowed.")

    # ------------------------------------------------------------------ #
    # eval() caching in quantised weights (if implemented).              #
    # ------------------------------------------------------------------ #
    def test_eval_weight_cache(self):
        block = self._make_block()
        x = torch.randn(1, 16, 32, 32)
        block.eval()
        with torch.no_grad():
            _ = block(x)  # first pass populates caches
            _ = block(x)  # second pass must not raise


if __name__ == "__main__":
    unittest.main()
