import unittest
import torch

from model import Conv2d                  # adjust import if layout differs
from utils import SymQuant8bit


class TestConv2d(unittest.TestCase):
    def _layer(self, in_c=8, out_c=8, k=3, q_en=True, out_int=False):
        q = SymQuant8bit(enabled=q_en)
        return Conv2d(in_c, out_c, k, padding=k // 2, quantizer=q, out_int=out_int)

    # ---------------------------------------------------------- #
    # Forward shape & dtype.                                     #
    # ---------------------------------------------------------- #
    def test_forward_shape(self):
        conv = self._layer()
        x = torch.randn(4, 8, 32, 32)
        y = conv(x)
        self.assertEqual(y.shape, (4, 8, 32, 32))

    # ---------------------------------------------------------- #
    # No-quant float path.                                       #
    # ---------------------------------------------------------- #
    def test_no_quant_path(self):
        conv = self._layer(q_en=False)
        x = torch.randn(2, 8, 16, 16)
        y = conv(x)
        self.assertEqual(y.shape, (2, 8, 16, 16))

    # ---------------------------------------------------------- #
    # Eval cache reuse.                                          #
    # ---------------------------------------------------------- #
    def test_eval_cache(self):
        conv = self._layer(out_int=True)   # cache path exercises int8→float
        conv.eval()
        x = torch.randn(1, 8, 10, 10)
        with torch.no_grad():
            _ = conv(x)   # first call fills caches
            _ = conv(x)   # reuse without error

    # ---------------------------------------------------------- #
    # Gradient propagation.                                      #
    # ---------------------------------------------------------- #
    def test_gradients(self):
        conv = self._layer()
        conv.train()
        x = torch.randn(1, 8, 20, 20, requires_grad=True)
        y = conv(x).sum()
        y.backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
