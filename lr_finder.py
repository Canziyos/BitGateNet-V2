"""
lr_finder.py.
Sweeps learning rate from 1e-5 to 1 over 100 mini-batches and logs loss.
Pick LR just before loss blows up.
"""

from __future__ import annotations
import os, glob, random, math, soundfile as sf, torch, torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from model import BitGateNetV2

data_dir = "dataset"
classes  = ["go", "stop", "_unknown_", "_silence_"]
batch_size = 64
steps = 100
seed = 7
torch.manual_seed(seed); random.seed(seed)


class AudioFolder(Dataset):
    def __init__(self, root: str):
        self.samples = [(p, lid)
            for lid, cls in enumerate(classes)
            for p in glob.glob(os.path.join(root, "train", cls, "**", "*.wav"), recursive=True)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        wav, _ = sf.read(path, dtype="float32")
        wav = torch.from_numpy(wav).unsqueeze(0)
        return wav, lbl

mel = torchaudio.transforms.MelSpectrogram(8000, n_mels=40, hop_length=128)
def collate(batch):
    x, y = [], []
    for wav, lbl in batch:
        m = mel(torchaudio.functional.resample(wav, 16000, 8000)).log2().clamp(min=-10)
        m = torch.nn.functional.pad(m, (0, 63 - m.size(-1))) if m.size(-1) < 63 else m[..., :63]
        x.append(m); y.append(lbl)
    return torch.stack(x), torch.tensor(y, dtype=torch.long)

dl = DataLoader(AudioFolder(data_dir), batch_size, shuffle=True, collate_fn=collate)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = BitGateNetV2(len(classes), q_en=False).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-5)

lrs, losses = [], []
lr_mult = (1.0 / 1e-5) ** (1 / steps)  # exponential step.

for i, (x, y) in zip(range(steps), dl):
    x, y = x.to(device), y.to(device)
    opt.param_groups[0]["lr"] *= lr_mult
    lr_now = opt.param_groups[0]["lr"]; lrs.append(lr_now)
    opt.zero_grad(); loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
    losses.append(loss.item())
    print(f"{i+1:03d}/{steps}  lr={lr_now:.2e}  loss={loss.item():.3f}")
    if math.isnan(loss.item()) or loss.item() > 4 * losses[0]:
        break

plt.plot(lrs, losses); plt.xscale("log"); plt.xlabel("LR"); plt.ylabel("Loss")
plt.title("LR Finder"); plt.savefig("lr_finder.png"); print("Saved lr_finder.png.")
