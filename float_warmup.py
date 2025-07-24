"""
float_warmup_qat.py
Stage A: 10 epochs float training with cosine LR schedule.
Stage B: enable QAT (quantscale=0.75) and fine-tune 3 epochs.
Prints confusion matrix at the end.
"""

from __future__ import annotations
import os, glob, random, soundfile as sf, torch, torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from model import BitGateNetV2

# ---------------- Config ---------------- #
data_dir = "dataset"
classes = ["go", "stop", "_unknown_", "_silence_"]
sample_rate_src, sample_rate_dst = 16_000, 8_000
batch_size = 64
epock_float, epochs_QAT = 10, 3
lr_float, lr_qat = 1e-3, 1e-4
seed = 7
torch.manual_seed(seed)
random.seed(seed)

# ---------------- Dataset ----------------#
class AudioFolder(Dataset):
    """Dataset: dataset/{split}/{class}/*.wav."""
    def __init__(self, root: str, split: str):
        self.samples = []
        for lid, cls in enumerate(classes):
            pattern = os.path.join(root, split, cls, "**", "*.wav")
            for p in glob.glob(pattern, recursive=True):
                self.samples.append((p, lid))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        wav, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(wav).unsqueeze(0)
        if sr != sample_rate_src:
            wav = torchaudio.functional.resample(wav, sr, sample_rate_src)
        return wav, lbl, path

# ---------------- Collate ---------------- #
mel = torchaudio.transforms.MelSpectrogram(sample_rate_dst, n_mels=40, hop_length=128)
FIX_T = 63

def collate(batch):
    xs, ys = [], []
    for wav, lbl, _ in batch:
        wav = torchaudio.functional.resample(wav, sample_rate_src, sample_rate_dst)
        m = mel(wav).log2().clamp(min=-10)
        if m.size(-1) < FIX_T:
            m = torch.nn.functional.pad(m, (0, FIX_T - m.size(-1)))
        else:
            s = (m.size(-1) - FIX_T) // 2
            m = m[..., s:s + FIX_T]
        xs.append(m)
        ys.append(lbl)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

# ---------------- Training helper ---------------- #
def run_epoch(model, loader, optimizer=None):
    """Run one epoch. If optimizer is None → validation only."""
    train_mode = optimizer is not None
    model.train(train_mode)
    correct, total = 0, 0
    all_p, all_l = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train_mode:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            all_p.append(pred.cpu())
            all_l.append(y.cpu())
    return 100 * correct / total, all_p, all_l

# -------------------------------------------------------------------------- #
def main():
    # DataLoaders
    train_dl = DataLoader(AudioFolder(data_dir, "train"), batch_size, shuffle=True, collate_fn=collate, num_workers=4)
    val_dl = DataLoader(AudioFolder(data_dir, "val"),   batch_size, shuffle=False, collate_fn=collate, num_workers=4)

    # Stage A: Float training.
    model = BitGateNetV2(len(classes), q_en=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr_float, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epock_float)

    print("Stage A: Float warm-up")
    for epoch in range(1, epock_float + 1):
        acc, _, _ = run_epoch(model, train_dl, opt)
        val_acc, _, _ = run_epoch(model, val_dl)
        print(f"Epoch {epoch}/{epock_float + epochs_QAT}: val_acc={val_acc:.2f}%")
        sched.step()

    # Stage B: QAT fine-tune.
    print("Stage B: QAT fine-tune")
    # Enable fake quant
    model.q_en = True

    for m in model.modules():
        if hasattr(m, "quantizer"):
            m.quantizer.enabled = True
            m.quantizer.quantscale = 0.75

    opt = torch.optim.Adam(model.parameters(), lr=lr_qat, weight_decay=1e-4)
    for epoch in range(epock_float + 1, epock_float + epochs_QAT + 1):
        acc, _, _ = run_epoch(model, train_dl, opt)
        val_acc, all_p, all_l = run_epoch(model, val_dl)
        print(f"Epoch {epoch}/{epock_float + epochs_QAT}: val_acc={val_acc:.2f}%")

    # Final confusion matrix.
    preds = torch.cat(all_p)
    labels = torch.cat(all_l)
    print(confusion_matrix(labels, preds))
    print(classification_report(labels, preds, target_names=classes, digits=2))

# ----------------------------------------- #
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # Windows for DataLoader workers.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main()
