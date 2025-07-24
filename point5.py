"""
train_quant_compare.py
Runs two short training sessions:
1) Float mode (q_en=False)
2) Quantized mode (q_en=True)
Compares their validation accuracies to see quantization impact.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import BitGateNetV2
from train import AudioFolder, collate, classes, dataset_dir, batch_size, n_worker

# ------------------------------------------------------------------ #
# Shared training function.                                          #
# ------------------------------------------------------------------ #
def run_training(mode_quant=False, epochs=1, lr=3e-4):
    """Train + validate once, return accuracy."""
    # Dataset & loaders
    train_ds = AudioFolder(dataset_dir, "train")
    val_ds   = AudioFolder(dataset_dir, "val")

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          collate_fn=collate, num_workers=n_worker)
    val_dl   = DataLoader(val_ds, batch_size, shuffle=False,
                          collate_fn=collate, num_workers=n_worker)

    # Model
    model = BitGateNetV2(num_classes=len(classes), q_en=mode_quant)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop (1 epoch)
    model.train()
    for x, y in tqdm(train_dl, desc=f"Training {'Quant' if mode_quant else 'Float'}", ncols=80):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()

    return 100.0 * correct / total

# ------------------------------------------------------------------ #
# Main.                                                              #
# ------------------------------------------------------------------ #
def main():
    acc_float = run_training(mode_quant=False)
    acc_quant = run_training(mode_quant=True)

    print(f"\nFloat accuracy: {acc_float:.2f}%")
    print(f"Quant accuracy: {acc_quant:.2f}%")
    print(f"Accuracy drop due to quantization: {acc_float - acc_quant:.2f}%")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()

# python train_quant_compare.py