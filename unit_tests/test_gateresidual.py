import unittest
import torch

from model import GateResidual           # adjust import if layout differs
from utils import SymQuant8bit


class TestGateResidual(unittest.TestCase):
    def _block(self, ch=32, depthwise=False, q_en=True):
        q = SymQuant8bit(enabled=q_en)
        return GateResidual(ch, ch // 2, ch, quantizer=q, depthwise=depthwise)

    # ---------------------------------------------------------- #
    # Forward shape.                                             #
    # ---------------------------------------------------------- #
    def test_forward_shape(self):
        gr = self._block()
        x = torch.randn(3, 32, 31, 29)     # odd H,W to stress upsample sizing
        y = gr(x)
        self.assertEqual(y.shape, x.shape)

    # ---------------------------------------------------------- #
    # Depth-wise option path.                                    #
    # ---------------------------------------------------------- #
    def test_depthwise_forward(self):
        gr = self._block(depthwise=True)
        x = torch.randn(2, 32, 24, 24)
        y = gr(x)
        self.assertEqual(y.shape, x.shape)

    # ---------------------------------------------------------- #
    # Gradient flow.                                             #
    # ---------------------------------------------------------- #
    def test_gradients(self):
        gr = self._block()
        gr.train()
        x = torch.randn(1, 32, 16, 16, requires_grad=True)
        y = gr(x).sum()
        y.backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
