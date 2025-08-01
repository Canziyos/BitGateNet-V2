import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf

from model import BitGateNetV2

# -------------------
# Config
# -------------------
classes = ["go", "stop", "other"]
checkpoint_path = "checkpoints/bitgatenet_wide.pth" 
audio_path = "test_audio.wav" 
sample_rate_src = 16_000   # original audio
sample_rate_dst = 8_000    # resample target
fix_frames = 63
n_mel = 40
mel_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate_dst,
    n_mels=n_mel,
    hop_length=128
)
# -------------------
# 1. Load model
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BitGateNetV2(num_classes=len(classes), q_en=False).to(device)

ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# -------------------
# 2. Preprocess audio
# -------------------
def preprocess_wav(path):
    # Load wav
    wav_np, sr = sf.read(path, dtype="float32")

    # Ensure channel-first tensor
    if wav_np.ndim == 1:
        wav_np = wav_np[:, None]
    wav = torch.from_numpy(wav_np.T)  # shape [1, N]

    # Resample to 8k
    wav = torchaudio.functional.resample(wav, sr, sample_rate_dst)

    # Mel spectrogram
    mel = mel_fn(wav)

    # Log compression + normalization
    mel = torch.log10(mel + 1e-6).clamp(min=-4)
    mel = (mel - mel.mean()) / (mel.std() + 1e-5)

    # Pad or crop to 63 frames
    if mel.size(-1) < fix_frames:
        mel = F.pad(mel, (0, fix_frames - mel.size(-1)))
    else:
        mel = mel[..., -fix_frames:]

    return mel  # shape [1, 40, 63]

# -------------------
# 3. Inference
# -------------------
mel = preprocess_wav(audio_path).unsqueeze(0).to(device)  # add batch dim

with torch.no_grad():
    logits = model(mel)
    pred_idx = logits.argmax(1).item()
    print(f"Predicted class: {classes[pred_idx]}")
