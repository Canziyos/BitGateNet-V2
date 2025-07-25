"""
overfit_balanced.py.
Proves BitGateNetV2 can memorise a tiny, class-balanced slice.
"""

from __future__ import annotations
import os, random, glob, soundfile as sf, torch, torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from collections import defaultdict
from tqdm import tqdm

from model import BitGateNetV2

# ------------ config ------------------------- #
dataset_dir = "dataset"
classes = ["go", "stop", "_unknown_", "_silence_"]
sample_rate_src, sample_rate_dst = 16_000, 8_000
n_per_class = 50
batch_size, epochs, lr = 64, 25, 1e-3
seed = 7
torch.manual_seed(seed); random.seed(seed)

# ------------ data loader -------- #
class AudioFolder(Dataset):
    def __init__(self, root: str, split: str):
        self.samples = []
        for lid, cls in enumerate(classes):
            pattern = os.path.join(root, split, cls, "**", "*.wav")
            for wav in glob.glob(pattern, recursive=True):
                self.samples.append((wav, lid))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        wav, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(wav).unsqueeze(0)
        if sr != sample_rate_src:
            wav = torchaudio.functional.resample(wav, sr, sample_rate_src)
        return wav, lbl

def build_subset(ds):
    idx_map = defaultdict(list)
    for i, (_, lbl) in enumerate(ds):
        idx_map[lbl].append(i)
    small = []
    for lbl in range(len(classes)):
        small += random.sample(idx_map[lbl], n_per_class)
    return Subset(ds, small)

mel = torchaudio.transforms.MelSpectrogram(sample_rate_dst, n_mels=40, hop_length=128)

FIX_T = 63
def collate(batch):
    xs, ys = [], []
    for wav, lbl in batch:
        wav = torchaudio.functional.resample(wav, sample_rate_src, sample_rate_dst)
        m = mel(wav).log2().clamp(min=-10)
        if m.size(-1) < FIX_T:
            m = torch.nn.functional.pad(m, (0, FIX_T - m.size(-1)))
        else:
            s = (m.size(-1) - FIX_T) // 2
            m = m[..., s:s + FIX_T]
        xs.append(m); ys.append(lbl)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

# ------------------------------------------------------------------ #
def main():
    train_base = AudioFolder(dataset_dir, "train")
    val_base   = AudioFolder(dataset_dir, "val")
    train_ds = build_subset(train_base)
    val_ds   = build_subset(val_base)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size, shuffle=False, collate_fn=collate)

    model = BitGateNetV2(num_classes=len(classes), q_en=False)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device)

    for epoch in range(epochs):
        model.train()

        for x, y in tqdm(train_dl, desc="Training", ncols=80):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc="Training", ncols=80):
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item(); total += y.numel()
        acc = 100 * correct / total
        print(f"Epoch {epoch+1:02d}/{epochs}: val_acc={acc:.2f}%")
        if acc > 95:
            print("Network can over-fit the tiny set.")
            break

if __name__ == "__main__":
    main()
