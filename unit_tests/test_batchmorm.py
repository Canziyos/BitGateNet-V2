import unittest
import torch

from model import BatchNorm2D 
from utils import SymQuant8bit


class TestBatchNorm2D(unittest.TestCase):
    def _layer(self, ch=8, q_en=True):
        quant = SymQuant8bit(enabled=q_en)
        return BatchNorm2D(ch, quantizer=quant)

    # ------------------------------------------------------------ #
    # Forward shape & dtype.                                       #
    # ------------------------------------------------------------ #
    def test_forward_shape(self):
        bn = self._layer()
        x = torch.randn(4, 8, 16, 16)
        y = bn(x)
        self.assertEqual(y.shape, (4, 8, 16, 16))

    # ------------------------------------------------------------ #
    # No-quant path.                                               #
    # ------------------------------------------------------------ #
    def test_no_quant_path(self):
        bn = self._layer(q_en=False)
        x = torch.randn(2, 8, 8, 8)
        y = bn(x)
        self.assertEqual(y.shape, (2, 8, 8, 8))

    # ------------------------------------------------------------ #
    # Eval cache reuse.                                            #
    # ------------------------------------------------------------ #
    def test_eval_cache(self):
        bn = self._layer()
        bn.eval()
        x = torch.randn(1, 8, 4, 4)
        with torch.no_grad():
            _ = bn(x)  # cache gamma/beta
            _ = bn(x)  # reuse without error

    # ------------------------------------------------------------ #
    # Gradient flow.                                               #
    # ------------------------------------------------------------ #
    def test_gradients(self):
        bn = self._layer()
        bn.train()
        x = torch.randn(3, 8, 5, 5, requires_grad=True)
        y = bn(x).sum()
        y.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
