# train_no_args.py — BitGateNetV2 full-run trainer
# Dataset structure: dataset/{train,val,test}/{class}

from __future__ import annotations
import os, random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
from dataset import AudioFolder, collate
from model import BitGateNetV2

# 1. Hyperparameters 
# ------------------ #
data_root = "dataset"
epochs_float = 100
epochs_qat = 0
batch_size = 64
workers = 4
seed  = 42
save_dir = "checkpoints"
user_amp = False           # set True if using mixed precision.
parience = 10              # early stopping patience (0 to disable).


random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 3. Dataset helpers.
# --------------------------- #
classes = ["go", "stop", "other"]


# 4. Checkpoint helpers.
# ------------------------------------------------- #
def save_ckpt(path: str, epoch: int, model, opt, sched, scaler,
              best_acc: float, hparams: dict):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "opt":   opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "best_acc": best_acc,
        "hparams":  hparams,
    }, path)
    print(f"Saved checkpoint => {path}")

# 5. Train / Val loop
# ------------------------------------------------------------ #
def run_epoch(model, loader, device, scaler, use_amp, train=False, opt=None, sched=None):
    model.train() if train else model.eval()
    total_loss = correct = total = 0.0
    phase = "Train" if train else "Val"
    for x, y in tqdm(loader, desc=phase, ncols=80, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        ctx = torch.cuda.amp.autocast() if use_amp else nullcontext()
        with ctx:
            logits = model(x)
            loss   = F.cross_entropy(logits, y)

        if train:
            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()

    return total_loss / len(loader), 100. * correct / total


# 6. Main routine
# --------------------------------------- #
def main():
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = user_amp and torch.cuda.is_available()

    # Data.
    train_ds = AudioFolder(data_root, "train", classes)
    val_ds = AudioFolder(data_root, "val", classes)
    print("Class counts:", {classes[k]: v for k, v in Counter(lbl for _, lbl in train_ds).items()})
    print(f"Train {len(train_ds)} | Val {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          collate_fn=collate, num_workers=workers,
                          pin_memory=use_amp, persistent_workers=use_amp)
    val_dl   = DataLoader(val_ds, batch_size, shuffle=False,
                          collate_fn=collate, num_workers=workers,
                          pin_memory=use_amp, persistent_workers=use_amp)

    # Model.
    model = BitGateNetV2(num_classes=len(classes), q_en=False).to(device)

    steps_pe = len(train_dl)
    total_steps_f = epochs_float * steps_pe
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps_f,
                                                pct_start=0.1, final_div_factor=100)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    best_acc = 0.0
    no_improve = 0


    for epoch in range(epochs_float):
        print(f"\n[Float {epoch+1}/{epochs_float}]")

        run_epoch(model, train_dl, device, scaler, use_amp, train=True, opt=opt, sched=sched)
        val_loss, val_acc = run_epoch(model, val_dl, device, scaler, use_amp)

        print(f"val_acc {val_acc:.2f}% | val_loss {val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            save_ckpt(Path(save_dir) / "best_float.pth", epoch, model, opt, sched, scaler, best_acc, {})
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0:
            save_ckpt(Path(save_dir) / f"float_ep{epoch+1}.pth", epoch, model, opt, sched, scaler, best_acc, {})

        if parience and no_improve >= parience:
            print(f"Early-stopping: no val-acc improvement in {parience} epochs.")
            break

    # QAT fine-tuning.
    if best_acc >= 75.0 and epochs_qat > 0:
        print(f"\nSwitching to QAT — best float acc {best_acc:.2f}%")
        qat_model = BitGateNetV2(num_classes=len(classes), q_en=True, quantscale=0.8).to(device)
        qat_model.load_state_dict(model.state_dict())
        model = qat_model

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        total_steps_q = epochs_qat * steps_pe
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps_q,
                                                    pct_start=0.1, final_div_factor=100)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        for epoch in range(epochs_qat):
            print(f"\n[QAT {epoch+1}/{epochs_qat}]")
            run_epoch(model, train_dl, device, scaler, use_amp, train=True, opt=opt, sched=sched)
            val_loss, val_acc = run_epoch(model, val_dl, device, scaler, use_amp)
            print(f"QAT val_acc {val_acc:.2f}% | val_loss {val_loss:.4f}")

            save_ckpt(Path(save_dir) / f"qat_ep{epoch+1}.pth", epoch, model, opt, sched, scaler, best_acc, {})

    print("Training complete.")

if __name__ == "__main__":
    main()
