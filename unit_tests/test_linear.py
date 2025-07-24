import unittest
import torch
from model import Linear
from utils import SymQuant8bit


class TestLinear(unittest.TestCase):
    def _layer(self, in_f=32, out_f=16, q_en=True, out_int=False):
        quant = SymQuant8bit(enabled=q_en)
        return Linear(in_f, out_f, bias=True, quantizer=quant, out_int=out_int)

    def test_forward_shape(self):
        layer = self._layer()
        x = torch.randn(4, 32)
        y = layer(x)
        self.assertEqual(y.shape, (4, 16))

    def test_no_quant_path(self):
        layer = self._layer(q_en=False)
        x = torch.randn(2, 32)
        y = layer(x)
        self.assertEqual(y.shape, (2, 16))

    def test_eval_cache_float(self):
        layer = self._layer(out_int=False)
        layer.eval()
        x = torch.randn(1, 32)
        with torch.no_grad():
            _ = layer(x)     # first pass caches
            _ = layer(x)     # reuse

    def test_eval_cache_int8(self):
        layer = self._layer(out_int=True)
        layer.eval()
        x = torch.randn(1, 32)
        with torch.no_grad():
            _ = layer(x)
            _ = layer(x)

    def test_gradients(self):
        layer = self._layer()
        layer.train()
        x = torch.randn(3, 32, requires_grad=True)
        y = layer(x).sum()
        y.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":   # noqa: D401
    unittest.main()
