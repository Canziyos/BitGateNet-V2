"""
train.py with progress bars and verbose training info.
"""

from __future__ import annotations
import os, glob, random
import torch, torchaudio, soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from model import BitGateNetV2

# ------------------------------------------------------------------ #
# Config.                                                            #
# ------------------------------------------------------------------ #
DATASET_DIR             = "dataset"
SAMPLE_RATE_SRC         = 16_000
SAMPLE_RATE_DST         = 8_000
BATCH_SIZE, LR          = 64, 3e-4
EPOCHS                  = 1
NUM_WORKERS             = 4
SEED                    = 42

CLASSES = ["go", "stop", "_unknown_", "_silence_"]

random.seed(SEED); torch.manual_seed(SEED)

# ------------------------------------------------------------------ #
# Dataset.                                                           #
# ------------------------------------------------------------------ #
class AudioFolder(Dataset):
    """Walks dataset/{split}/{class}/*.wav and returns waveform, label_id."""
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

# ------------------------------------------------------------------ #
# Collate => log-Mel batch.                                         #
# ------------------------------------------------------------------ #
mel_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE_DST, n_mels=40, hop_length=128
)

FIX_FRAMES = 63

def collate(batch):
    wavs, labels = [], []
    for wav, label in batch:
        wav = torchaudio.functional.resample(wav, SAMPLE_RATE_SRC, SAMPLE_RATE_DST)
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

# ------------------------------------------------------------------ #
# Main.                                                              #
# ------------------------------------------------------------------ #
def main():
    train_ds = AudioFolder(DATASET_DIR, "train")
    val_ds   = AudioFolder(DATASET_DIR, "val")

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    train_dl = DataLoader(train_ds, BATCH_SIZE, True,
                          collate_fn=collate, num_workers=NUM_WORKERS)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, False,
                          collate_fn=collate, num_workers=NUM_WORKERS)

    model = BitGateNetV2(num_classes=len(CLASSES), q_en=False, quantscale=1.0)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        # ---------- train ---------- #
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_dl, desc="Training", ncols=80):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dl)
        print(f"Training loss: {avg_train_loss:.4f}")

        # ---------- val ------------ #
        model.eval(); correct = total = vloss = 0.0
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

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
