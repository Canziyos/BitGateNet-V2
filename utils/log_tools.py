import logging
import os
import numpy as np
import torch

# Global toggle to enable/disable binary tensor logging
LOG_TENSOR_ENABLED = False  # Set to True if binary logs are needed.

class LoggerUnit:
    def __init__(self, name, level=logging.DEBUG):  # Set default level.
        self.logger = logging.getLogger(name)
        self.level = level
        self.setup_logger()

    def setup_logger(self):
        self.logger.setLevel(self.level)

        if not any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)  # Also set handler.

            formatter = logging.Formatter(
                "%(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)

        self.logger.propagate = False

    def get_logger(self):
        return self.logger


def log_tensor(name: str, tensor: torch.Tensor, outdir: str = "logs", verbose: bool = False):
    """
    Save a tensor to a .bin file with shape info (for embedded C verification).

    Args:
        name (str): File name (without extension).
        tensor (torch.Tensor): The tensor to log.
        outdir (str): Output folder.
        verbose (bool): If True, prints shape and path info.
    """
    if not LOG_TENSOR_ENABLED:
        return

    tensor = tensor.detach().cpu().numpy()

    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, f"{name}.bin")

    with open(filepath, 'wb') as f:
        for dim in tensor.shape:
            f.write(np.array(dim, dtype=np.float32).tobytes())
        f.write(tensor.flatten().astype(np.float32).tobytes())

    if verbose:
        print(f"[log_tensor] Saved: {filepath} | Shape: {tensor.shape}")
