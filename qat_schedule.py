"""
qat_schedule.py.
Stage A: 10 epochs float.
Stage B: enable fake-quant (quantscale=0.75) and fine-tune 3 epochs.
"""

from __future__ import annotations
import os, glob, random, soundfile as sf, torch, torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import BitGateNetV2

data_dir = "dataset"
classes = ["go", "stop", "_unknown_", "_silence_"]
sample_rate_src, sample_rate_dst = 16_000, 8_000
batch_size = 64
epock_float, epochs_qat = 10, 3
lr_float, lr_qat = 1e-3, 1e-4
seed = 7
torch.manual_seed(seed); random.seed(seed)

class AudioFolder(Dataset):
    def __init__(self, root: str, split: str):
        self.samples = [(p, lid)
            for lid, cls in enumerate(classes)
            for p in glob.glob(os.path.join(root, split, cls, "**", "*.wav"), recursive=True)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        wav, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(wav).unsqueeze(0)
        if sr != sample_rate_src:
            wav = torchaudio.functional.resample(wav, sr, sample_rate_src)
        return wav, lbl

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
            s = (m.size(-1) - FIX_T) // 2; m = m[..., s:s+FIX_T]
        xs.append(m); ys.append(lbl)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

train_dl = DataLoader(AudioFolder(data_dir, "train"), batch_size, True,  collate_fn=collate, num_workers=4)
val_dl   = DataLoader(AudioFolder(data_dir, "val"),   batch_size, False, collate_fn=collate, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = BitGateNetV2(len(classes), q_en=False).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=lr_float)
sched  = torch.optim.CosineAnnealingLR(opt, T_max=epock_float)

def run_epoch(epoch_i, n_epochs):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item(); total += y.numel()
    acc = 100 * correct / total
    print(f"Epoch {epoch_i}/{n_epochs}: val_acc={acc:.2f}%")

print("Stage A: float training.")
for e in range(1, epock_float + 1):
    run_epoch(e, epock_float + epochs_qat)
    sched.step()

print("Stage B: enable QAT.")
model.q_en = True
for m in model.modules():
    if hasattr(m, "quantizer"):
        m.quantizer.enabled = True
opt = torch.optim.Adam(model.parameters(), lr=lr_qat)
for e in range(epock_float + 1, epock_float + epochs_qat + 1):
    run_epoch(e, epock_float + epochs_qat)
