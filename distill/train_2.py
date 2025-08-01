import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from .extended_loss import kd_feature_loss
from tqdm import tqdm
import os

# -------------------------------
# Validation helper (same as before)
# -------------------------------
@torch.no_grad()
def validate(model, val_dl, device):
    model.eval()
    loss_sum = correct = total = 0

    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # Only need logits for val
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    acc = 100 * correct / total
    loss = loss_sum / total
    print(f"[VAL] acc {acc:.2f}% | loss {loss:.4f}")
    model.train()
    return acc, loss


# -------------------------------
# Feature KD Training Loop with Projection
# -------------------------------
def distill_feature_train(student,
                          teacher,
                          train_dl,
                          val_dl,
                          optimizer,
                          scheduler,
                          epochs=20,
                          alpha=0.5,
                          temperature=2.0,
                          beta=0.3,
                          patience=5,
                          device="cpu"):

    teacher.eval()  # freeze teacher
    scaler = GradScaler(enabled=device.startswith("cuda"))
    best_acc = 0.0
    no_improve = 0   # track epochs without improvement
    os.makedirs("checkpoints", exist_ok=True)

    # ---------------- Projection layer ----------------
    # Will map student features → teacher feature channel size
    proj = None

    for epoch in range(1, epochs + 1):
        student.train()
        running_loss = running_correct = running_total = 0

        pbar = tqdm(train_dl, ncols=90, desc=f"[FeatureKD E{epoch:02d}/{epochs}]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            # Forward teacher (features + logits)
            with torch.no_grad():
                teacher_logits, teacher_features = teacher(x, return_features=True)

            # Forward student (features + logits)
            with autocast(enabled=device.startswith("cuda")):
                student_logits, student_features = student(x, return_features=True)

                # Initialize projection layer dynamically on first pass
                if proj is None and student_features.shape[1] != teacher_features.shape[1]:
                    proj = torch.nn.Conv2d(student_features.shape[1],
                                           teacher_features.shape[1],
                                           kernel_size=1).to(device)

                # If projection exists, map student features
                if proj is not None:
                    student_features = proj(student_features)

                # Combined CE + KL + MSE feature loss
                loss = kd_feature_loss(student_logits,
                                       teacher_logits,
                                       y,
                                       student_features,
                                       teacher_features,
                                       alpha=alpha,
                                       temperature=temperature,
                                       beta=beta)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_correct += (student_logits.argmax(1) == y).sum().item()
            running_total += y.size(0)

            pbar.set_postfix(loss=f"{running_loss/len(pbar):.4f}",
                             acc=f"{100*running_correct/running_total:.2f}%")

        # ---------------- Validation ----------------
        val_acc, _ = validate(student, val_dl, device)
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0  # reset early stopping counter
            ckpt_path = "checkpoints/student_kd_feature_best.pth"
            torch.save({
                "epoch": epoch,
                "model": student.state_dict(),
                "opt": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "best_acc": best_acc
            }, ckpt_path)
            print(f"New best saved => {ckpt_path} ({best_acc:.2f}%)")
        else:
            no_improve += 1

        # Early stopping check
        if patience and no_improve >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

        scheduler.step()

    print(f"Finished Feature KD. Best val acc: {best_acc:.2f}%")
