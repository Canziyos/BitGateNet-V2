"""
debug_forward.py
Focused sanity check for BitGateNetV2:
- Overfit one batch (check label/math correctness).
- Includes data inspection (print labels + visualize spectrogram).
"""

import os, glob, random
import torch, torchaudio, soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import BitGateNetV2

# -------------------------------------------------- #
# Config.
# -------------------------------------------------- #
DATASET_DIR      = "dataset"
SAMPLE_RATE_SRC  = 16_000
SAMPLE_RATE_DST  = 8_000
BATCH_SIZE       = 8      # small batch for debugging
FIX_FRAMES       = 63
SEED             = 42
CLASSES = ["go", "stop", "_unknown_", "_silence_"]

random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------------------------- #
# Dataset + Collate
# -------------------------------------------------- #
class AudioFolder(Dataset):
    def __init__(self, root: str, split: str):
        self.samples: list[tuple[str,int]] = []
        for label_id, cls in enumerate(CLASSES):
            folder = os.path.join(root, split, cls)
            for wav in glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True):
                self.samples.append((wav, label_id))
        if not self.samples:
            raise RuntimeError(f"No wav files in {root}/{split}.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        wav_np, sr = sf.read(wav_path, dtype="float32")
        if wav_np.ndim == 1:
            wav_np = wav_np[:, None]
        wav = torch.from_numpy(wav_np.T)
        if sr != SAMPLE_RATE_SRC:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE_SRC)
        return wav, label

mel_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE_DST, n_mels=40, hop_length=128
)

def collate(batch):
    wavs, labels = [], []
    for wav, label in batch:
        wav = torchaudio.functional.resample(wav, SAMPLE_RATE_SRC, SAMPLE_RATE_DST)
        mel = mel_fn(wav).log2().clamp(min=-10)

        T = mel.size(-1)
        if T < FIX_FRAMES:
            mel = F.pad(mel, (0, FIX_FRAMES - T))
        else:
            mel = mel[..., -FIX_FRAMES:]

        wavs.append(mel)
        labels.append(label)

    x = torch.stack(wavs)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y

# -------------------------------------------------- #
# Test: Overfit one batch
# -------------------------------------------------- #
def test_overfit_one_batch(model, device):
    print("\n[Overfit One Batch Test]")
    train_ds = AudioFolder(DATASET_DIR, "train")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate)

    x, y = next(iter(train_dl))  # grab one batch
    x, y = x.to(device), y.to(device)

    # Debug conv1 in depth
    model.eval()
    with torch.no_grad():
        conv1_out = model.conv1(x)  # raw Conv1 output
        print("\n[Conv1 Output Stats]")
        print(f"Shape: {conv1_out.shape}")
        print(f"Min: {conv1_out.min().item():.4f}, Max: {conv1_out.max().item():.4f}")
        print(f"Mean: {conv1_out.mean().item():.4f}, Std: {conv1_out.std().item():.4f}")

    # --- Inspect batch before training --- (Debug)
    # print("\n[Data Inspection]")
    # print(f"Batch shape: {x.shape}")  # Expect: (B, 1, 40, 63)
    # print(f"Labels: {y.tolist()}")
    # class_names = [CLASSES[label] for label in y.tolist()]
    # print(f"Label names: {class_names}")
    # unique_labels = set(y.tolist())
    # print(f"Unique labels in this batch: {unique_labels} (count: {len(unique_labels)})")

    # # Visualize first sample’s spectrogram (Debug)
    # mel_example = x[0].squeeze(0).cpu().numpy()  # shape (40, 63)
    # plt.imshow(mel_example, aspect='auto', origin='lower')
    # plt.title(f"Mel Spectrogram (Label: {class_names[0]})")
    # plt.colorbar()
    # plt.show()

    # --- Training loop on single batch ---
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(200):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()

        if (step + 1) % 20 == 0:
            with torch.no_grad():
                pred = model(x).argmax(1)
                acc = (pred == y).float().mean().item() * 100
            print(f"Step {step+1:03d}: loss={loss.item():.4f}, acc={acc:.2f}%")

# -------------------------------------------------- #
# import torch.nn as nn
# # baseline model, replacing the bitgatenet itself.
# # uncomment model instantiating this model- below in main!
# class BaselineLinear(nn.Module):
#     """Simple baseline: flatten mel spectrogram -> Linear -> 4-class logits."""
#     def __init__(self, num_classes=4):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(40 * 63, num_classes)

#     def forward(self, x):
#         x = self.flatten(x)
#         return self.fc(x)
# ---------------------------------------------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BitGateNetV2(num_classes=len(CLASSES), q_en=False).to(device)
    # model = BaselineLinear(num_classes=len(CLASSES)).to(device)
    test_overfit_one_batch(model, device)
