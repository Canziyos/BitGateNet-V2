import unittest
import torch

from model import DualGateBN
from utils import SymQuant8bit


class TestDualGateBN(unittest.TestCase):
    def _block(self, ch=32, depthwise=False, q_en=True):
        q = SymQuant8bit(enabled=q_en)
        return DualGateBN(ch, ch // 2, ch, ch // 2, ch, quantizer=q, depthwise=depthwise)

    # ---------------------------------------------------------- #
    # Forward shape.                                             #
    # ---------------------------------------------------------- #
    def test_forward_shape(self):
        dgbn = self._block()
        x = torch.randn(2, 32, 28, 28)
        y = dgbn(x)
        self.assertEqual(y.shape, x.shape)

    # ---------------------------------------------------------- #
    # Depth-wise variant.                                        #
    # ---------------------------------------------------------- #
    def test_depthwise_forward(self):
        dgbn = self._block(depthwise=True)
        x = torch.randn(1, 32, 20, 20)
        y = dgbn(x)
        self.assertEqual(y.shape, x.shape)

    # ---------------------------------------------------------- #
    # No-quant float path.                                       #
    # ---------------------------------------------------------- #
    def test_no_quant_path(self):
        dgbn = self._block(q_en=False)
        x = torch.randn(1, 32, 16, 16)
        y = dgbn(x)
        self.assertEqual(y.shape, x.shape)

    # ---------------------------------------------------------- #
    # Gradient propagation.                                      #
    # ---------------------------------------------------------- #
    def test_gradients(self):
        dgbn = self._block()
        dgbn.train()
        x = torch.randn(1, 32, 18, 18, requires_grad=True)
        y = dgbn(x).sum()
        y.backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
