# data/dataset.py
"""
Dataset + preprocessing utilities for BitGateNet-V2.
Shared by training and testing scripts.
"""

import os, glob
import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F
from torch.utils.data import Dataset

# Configuration (fixed for dataset)
# ------------------------------------------------------------ #
sample_rate_src = 16_000   # original audio.
sample_rate_dst = 8_000    # resample target.
fix_frames = 63            # fixed spectrogram time frames
n_mel = 40


# Dataset: AudioFolder
# ------------------------------------------------------------ #
class AudioFolder(Dataset):
    """Walk dataset/{split}/{class} and return waveform, label."""
    def __init__(self, root: str, split: str, classes: list[str]):
        self.samples: list[tuple[str,int]] = []
        self.classes = classes

        for lid, cls in enumerate(classes):
            folder = os.path.join(root, split, cls)
            for wav in glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True):
                self.samples.append((wav, lid))

        if not self.samples:
            raise RuntimeError(f"No wav files found in {root}/{split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        wav_np, sr = sf.read(wav_path, dtype="float32")

        # ensure channel-first shape
        if wav_np.ndim == 1:
            wav_np = wav_np[:, None]

        wav = torch.from_numpy(wav_np.T)
        if sr != sample_rate_src:
            wav = torchaudio.functional.resample(wav, sr, sample_rate_src)
        return wav, label


# MelSpectrogram transform
# ------------------------------------------------------------ #
mel_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate_dst,
    n_mels=n_mel,
    hop_length=128
)


# Collate function: resample -> mel -> log compression -> normalize
# ----------------------------------------------------------------- #
def collate(batch):
    wavs, labels = [], []
    for wav, label in batch:
        # Resample to target rate.
        wav = torchaudio.functional.resample(wav, sample_rate_src, sample_rate_dst)

        # Convert to mel spectrogram.
        mel = mel_fn(wav)

        # Log compression + normalization.
        mel = torch.log10(mel + 1e-6).clamp(min=-4)
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)

        # Pad or crop to Fix_frame.
        if mel.size(-1) < fix_frames:
            mel = F.pad(mel, (0, fix_frames - mel.size(-1)))
        else:
            mel = mel[..., -fix_frames:]

        wavs.append(mel)
        labels.append(label)

    return torch.stack(wavs), torch.tensor(labels, dtype=torch.long)
