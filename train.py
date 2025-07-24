"""
train.py
Baseline training script with:
- Class count + duration debug print
- Standard training loop (quant enabled by default)
- Dataset structure: dataset/{train,val,test}/{class}
"""

from __future__ import annotations
import os, glob, random
import torch, torchaudio, soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import BitGateNetV2

# --------------------------------------------------#
dataset_dir = "dataset"
sample_rate_src = 16_000
sample_rate_dst = 8_000
batch_size = 64
lr = 3e-4
epocks = 1
n_worker = 4
seed = 42

classes = ["go", "stop", "_unknown_", "_silence_"]

random.seed(seed)
torch.manual_seed(seed)

# ----------------------------------------------------#
class AudioFolder(Dataset):
    """Walks dataset/{split}/{class}/*.wav and returns waveform, label_id."""
    def __init__(self, root: str, split: str):
        self.samples: list[tuple[str,int]] = []
        for label_id, cls in enumerate(classes):
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
        if sr != sample_rate_src:
            wav = torchaudio.functional.resample(wav, sr, sample_rate_src)
        return wav, label

# ------------------------------------------#
# Collate => log-Mel batch.         #
# -------------------------------------------#
mel_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate_dst, n_mels=40, hop_length=128)
FIX_FRAMES = 63

def collate(batch):
    wavs, labels = [], []
    for wav, label in batch:
        wav = torchaudio.functional.resample(wav, sample_rate_src, sample_rate_dst)
        mel = mel_fn(wav).log2().clamp(min=-10)

        T = mel.size(-1)
        if T < FIX_FRAMES:
            pad_amt = FIX_FRAMES - T
            mel = torch.nn.functional.pad(mel, (0, pad_amt))
        else:
            start = (T - FIX_FRAMES) // 2
            mel = mel[..., start : start + FIX_FRAMES]

        wavs.append(mel)
        labels.append(label)

    x = torch.stack(wavs)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def main():
    train_ds = AudioFolder(dataset_dir, "train")
    val_ds   = AudioFolder(dataset_dir, "val")

    # ---------------- Debug: Class counts & durations ---------------- #
    from collections import Counter
    import numpy as np

    cnt = Counter(lbl for _, lbl in train_ds)
    print("Train class counts:", {classes[k]: v for k, v in cnt.items()})

    dur = [sf.info(path).frames / sf.info(path).samplerate * 1000
           for path, _ in train_ds.samples]
    print("Clip duration  P50/P90/P99:", np.percentile(dur, [50, 90, 99]))

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # ---------------- DataLoaders ---------------- #
    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          collate_fn=collate, num_workers=n_worker)
    val_dl = DataLoader(val_ds,   batch_size, shuffle=False,
                          collate_fn=collate, num_workers=n_worker)

    # ---------------- Model + Optimizer ---------------- #
    model = BitGateNetV2(num_classes=len(classes), q_en=True, quantscale=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2)

    # ---------------- Training loop ------------- #
    for epoch in range(epocks):
        print(f"\n--- Epoch {epoch+1}/{epocks} ---")

        # Training.
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_dl, desc="Training", ncols=80):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Training loss: {total_loss/len(train_dl):.4f}")

        # Validation
        model.eval()
        correct = total = vloss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc="Validation", ncols=80):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vloss += F.cross_entropy(logits, y, reduction="sum").item()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.numel()

        acc = 100.0 * correct / total
        print(f"Validation loss: {vloss/total:.4f} | Accuracy: {acc:.2f}%")
        sched.step(vloss / total)

# -----------------------------#
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
