import torch, random, os, numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.data import AudioFolder, collate
from model import BitGateNetV2
#from .train import distill_train
from .train_2 import distill_feature_train 

# config (hard-coded for now).
# -------------------------------#
seed = 42
data_root = "dataset"
classes = ["go", "stop", "other"]

teacher_ckpt = "checkpoints/wider_plus_dep_head.pth"
epochs = 15
batch_size = 64
lr = 3e-4
alpha = 0.5
temperature = 2.0
beta = 0.3  
use_feature_kd = True   # toggle between KD types

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------#
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

# teacher (frozen).
# -------------------------------#
teacher = BitGateNetV2(num_classes=len(classes), q_en=False).to(device)
state   = torch.load(teacher_ckpt, map_location=device)
teacher.load_state_dict(state["model"])
teacher.eval()

# student.
# -------------------------------#
student = BitGateNetV2(num_classes=len(classes), q_en=False, width_mult=1.0).to(device)

# data.
# -------------------------------#
train_ds = AudioFolder(data_root, "train", classes)
val_ds = AudioFolder(data_root, "val",   classes)
train_dl = DataLoader(train_ds, batch_size, shuffle=True,  collate_fn=collate)
val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=collate)

# optim / sched.
# -------------------------------#
optimizer  = torch.optim.Adam(student.parameters(), lr=lr)
steps_per_epoch = len(train_dl)
scheduler  = CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)

# run KD (classic or feature).
# -------------------------------#
if use_feature_kd:
    distill_feature_train(student=student,
                          teacher=teacher,
                          train_dl=train_dl,
                          val_dl=val_dl,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          epochs=epochs,
                          alpha=alpha,
                          patience=5, 
                          temperature=temperature,
                          beta=beta,
    )