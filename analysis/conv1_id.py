"""
debug_conv1_identity.py
Focused debug: Conv1 output analysis with identity (delta) initialization.
Goal: See if Conv1 scaling/initialization prevents overfitting.
"""

import os, glob, random
import torch, torchaudio, soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import BitGateNetV2

# -------------------------------------------------- #
# Config
# -------------------------------------------------- #
DATASET_DIR      = "dataset"
SAMPLE_RATE_SRC  = 16_000
SAMPLE_RATE_DST  = 8_000
BATCH_SIZE       = 8
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
# Overfit Test (Conv1 identity focus)
# -------------------------------------------------- #
def test_conv1_identity(model, device):
    print("\n[Overfit One Batch Test: Conv1 Identity Init]")

    # Load batch
    train_ds = AudioFolder(DATASET_DIR, "train")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate)

    x, y = next(iter(train_dl))
    x, y = x.to(device), y.to(device)

    # ---- Identity initialization for Conv1 ----
    with torch.no_grad():
        model.conv1.weight.zero_()
        center = model.conv1.kernel_size[0] // 2
        for out_ch in range(model.conv1.out_channels):
            model.conv1.weight[out_ch, 0, center, center] = 1.0
        if model.conv1.bias is not None:
            model.conv1.bias.zero_()

    print("Conv1 initialized as identity (delta kernel).")

    # Print Conv1 weight stats
    print(f"Weight stats -> min: {model.conv1.weight.min():.4f}, "
          f"max: {model.conv1.weight.max():.4f}, "
          f"mean: {model.conv1.weight.mean():.4f}, "
          f"std: {model.conv1.weight.std():.4f}")

    # ---- Print Conv1 output stats ----
    model.eval()
    with torch.no_grad():
        conv1_out = model.conv1(x)
        print("\n[Conv1 Output Stats]")
        print(f"Shape: {conv1_out.shape}")
        print(f"Min: {conv1_out.min().item():.4f}, Max: {conv1_out.max().item():.4f}")
        print(f"Mean: {conv1_out.mean().item():.4f}, Std: {conv1_out.std().item():.4f}")

    # ---- Overfit loop ----
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
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BitGateNetV2(num_classes=len(CLASSES), q_en=False).to(device)
    test_conv1_identity(model, device)
