import unittest
import torch
from utils.symquant8bit import SymQuant8bit


class TestSymQuant8bit(unittest.TestCase):
    def _roundtrip_ok(self, x, q):
        q_x, scale = q.quantize(x)
        x_hat = q.dequantize(q_x, scale)
        max_abs_err = (x - x_hat).abs().max().item()
        max_abs_val = x.abs().max().item()
        bound = max_abs_val / 254.0 + 1e-6        # 1/2-step of an int8 bucket
        self.assertLessEqual(max_abs_err, bound, "Round-trip error too high.")

    # ---------------------- #
    # Fundamental behaviour. #
    # --------------------- #

    def test_scalar_roundtrip(self):
        x = torch.tensor(3.14159, dtype=torch.float32)
        q = SymQuant8bit(enabled=True)
        self._roundtrip_ok(x, q)

    def test_vector_roundtrip(self):
        x = torch.randn(256)
        q = SymQuant8bit(enabled=True)
        self._roundtrip_ok(x, q)

    def test_group_roundtrip(self):
        x = torch.randn(4, 16, 8, 8)      # NCHW
        q = SymQuant8bit(group_dim=1, group_size=4)
        self._roundtrip_ok(x, q)

    # ------------------ #
    # Operational flags. #
    # ------------------ #

    def test_disabled_passthrough(self):
        x = torch.randn(10)
        q = SymQuant8bit(enabled=False)
        q_x, scale = q.quantize(x)
        self.assertTrue(torch.allclose(x, q_x), "Disabled quantiser modified tensor.")
        self.assertEqual(scale, 1)

    def test_int8_export(self):
        x = torch.randn(128)
        q = SymQuant8bit(return_int=True)
        q_x, _ = q.quantize(x)
        self.assertEqual(q_x.dtype, torch.int8, "return_int flag did not yield int8 output.")

    # -------------- #
    # Gradient flow. #
    # -------------- #

    def test_ste_gradient(self):
        x = torch.randn(32, requires_grad=True)
        q = SymQuant8bit()
        q_x, scale = q.quantize(x)
        y = q.apply_fake_quant(q_x, scale, x).sum()
        y.backward()
        self.assertIsNotNone(x.grad, "No gradient passed through STE.")
        self.assertGreater(x.grad.abs().sum().item(), 0, "Gradient is zero everywhere.")

    # ----------------------------------------- #
    # Consistency across group_size edge-cases. #
    # ----------------------------------------- #

    def test_group_size_bigger_than_channels(self):
        x = torch.randn(1, 3, 4, 4)
        q = SymQuant8bit(group_dim=1, group_size=16)
        self._roundtrip_ok(x, q)


if __name__ == "__main__":
    unittest.main()
