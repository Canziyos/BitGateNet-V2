# test_compare.py — Compare two checkpoints of BitGateNetV2
# Dataset structure: dataset/{train,val,test}/{class}
from collections import Counter
import torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from dataset import AudioFolder, collate
from model import BitGateNetV2

# ------------------------------------------------------------ #
# configuration
# ------------------------------------------------------------ #
data_root = "dataset"
batch_size = 32
workers = 4
checkpoints = [
    "checkpoints/best_float.pth",
    "checkpoints/b_float.pth"
]
use_amp = False
classes = ["go", "stop", "other"]

# ------------------------------------------------------------ #
# evaluation routine
# ------------------------------------------------------------ #
def evaluate(ckpt_path, test_dl, device, amp_enabled):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BitGateNetV2(num_classes=len(classes), q_en=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        context = torch.cuda.amp.autocast() if amp_enabled else torch.autocast("cpu", enabled=False)
        with context:
            for x, y in tqdm(test_dl, desc=f"Testing {ckpt_path}", ncols=80):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                total_loss += F.cross_entropy(logits, y, reduction="sum").item()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.numel()

    acc = 100.0 * correct / total
    loss = total_loss / total
    print(f"\n{ckpt_path} -> Test Accuracy: {acc:.2f}% | Test Loss: {loss:.4f}")
    return acc, loss
def evaluate_with_confusion(ckpt_path, test_dl, device, amp_enabled):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BitGateNetV2(num_classes=len(classes), q_en=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        context = torch.cuda.amp.autocast() if amp_enabled else torch.autocast("cpu", enabled=False)
        with context:
            for x, y in tqdm(test_dl, desc=f"Testing {ckpt_path}", ncols=80):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                total_loss += F.cross_entropy(logits, y, reduction="sum").item()
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

    acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    loss = total_loss / len(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n{ckpt_path} -> Test Accuracy: {acc:.2f}% | Test Loss: {loss:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, cls in enumerate(classes):
        print(f"{cls}: {per_class_acc[i]*100:.2f}%")

    return cm, per_class_acc
# ------------------------------------------------------------ #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = use_amp and torch.cuda.is_available()

    test_ds = AudioFolder(data_root, "test", classes)
    print("Class counts (test):", {classes[k]: v for k, v in Counter(lbl for _, lbl in test_ds).items()})
    print(f"Total test samples: {len(test_ds)}")

    test_dl = DataLoader(test_ds, batch_size, shuffle=False,
                         collate_fn=collate, num_workers=workers,
                         pin_memory=amp_enabled, persistent_workers=amp_enabled)

    for ckpt_path in checkpoints:
        #evaluate(ckpt_path, test_dl, device, amp_enabled)
        evaluate_with_confusion(ckpt_path, test_dl, device, amp_enabled)
