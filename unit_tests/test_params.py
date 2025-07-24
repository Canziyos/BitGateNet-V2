import unittest
import torch
from model import Parameter
from utils import SymQuant8bit


class TestParameter(unittest.TestCase):
    def test_forward_standalone(self):
        p = Parameter(torch.tensor([3.0]))
        out = p()
        self.assertEqual(out.shape, (1,))
        self.assertTrue(out.requires_grad)
        self.assertTrue(out.dtype.is_floating_point)

    def test_forward_multiply(self):
        p = Parameter(torch.ones(4))
        x = torch.arange(4.0)
        y = p(x)
        self.assertTrue(torch.allclose(y, x))

    def test_quantised_eval_cache_float(self):
        p = Parameter(torch.randn(5), quantizer=SymQuant8bit(), return_int=False)
        p.eval()
        x = torch.randn(5)
        with torch.no_grad():
            _ = p(x)   # first call caches float tensor
            _ = p(x)   # second call reuses cache without error

    def test_quantised_eval_cache_int8(self):
        p = Parameter(torch.randn(5), quantizer=SymQuant8bit(return_int=True), return_int=True)
        p.eval()
        x = torch.randn(5)
        with torch.no_grad():
            _ = p(x)   # caches de-quantised float
            _ = p(x)   # second call reuses cache

    def test_gradient(self):
        p = Parameter(torch.tensor([2.0]), quantizer=SymQuant8bit(), return_int=False)
        p.train()
        x = torch.tensor([3.0], requires_grad=True)
        y = p(x).sum()
        y.backward()
        self.assertIsNotNone(p.param.grad)
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
