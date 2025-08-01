import unittest
import torch
from model import BitGateNetV2

class TestBitGateNetV2(unittest.TestCase):
    def setUp(self):
        self.num_classes = 4
        self.model_q = BitGateNetV2(
            num_classes=self.num_classes,
            quantscale=1.0,
            q_en=True,
            width_mult=1.0,
        ).eval()

    # -------------#
    # Forward pass.#
    # -------------#
    def test_forward_shape(self):
        x = torch.randn(3, 1, 32, 32)
        y = self.model_q(x)
        self.assertEqual(y.shape, (3, self.num_classes))
        self.assertTrue(y.dtype.is_floating_point)

    # ----------- #
    # Float path. #
    # ----------- #
    def test_forward_no_quant(self):
        model_f = BitGateNetV2(num_classes=self.num_classes, q_en=False)
        x = torch.randn(2, 1, 32, 32)
        y = model_f(x)
        self.assertEqual(y.shape, (2, self.num_classes))

    # --------- #
    # Gradients.#
    # ----------#
    def test_gradients(self):
        model = BitGateNetV2(num_classes=self.num_classes, q_en=True).train()
        x = torch.randn(1, 1, 32, 32, requires_grad=True)
        y = model(x).sum()
        y.backward()
        self.assertGreater(sum(p.grad.abs().sum() for p in model.parameters()).item(), 0.0)

    # ----------------- #
    # Eval cache reuse. #
    # ----------------- #
    def test_eval_cache(self):
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            _ = self.model_q(x)
            _ = self.model_q(x)   # second pass reuses cached tensors.

if __name__ == "__main__":
    unittest.main()
