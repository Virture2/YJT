import sys
import os
import torch
import math
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, '../')
from src.data.components.new_KITTI_dataset import KITTI
from src.utils import new_custom_transform
from src.models.components.new_vsvio import Encoder
from src.utils.kitti_utils import eulerAnglesToRotationMatrixTorch as etr
from src.utils import rpmg

class ObjFromDict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

params = ObjFromDict({
    "img_w": 512,
    "img_h": 256,
    "v_f_len": 512,
    "i_f_len": 256,
    "g_f_len": 128,
    "imu_dropout": 0.1,
    "seq_len": 11
})

class GNSSPoseNet(nn.Module):
    def __init__(self, encoder: Encoder, pose_dim=6):
        super().__init__()
        self.gnss_encoder = encoder.gnss_encoder
        self.pose_head = nn.Sequential(
            nn.Linear(params.g_f_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pose_dim)
        )

    def forward(self, gnss):
        feat_g = self.gnss_encoder(gnss)
        return self.pose_head(feat_g)

class RPMGPoseLoss(nn.Module):
    def __init__(self, angle_weight=100):
        super().__init__()
        self.angle_weight = angle_weight

    def forward(self, poses, gts):
        B, T, _ = poses.shape
        R_pred = etr(poses[:, :, :3]).view(B*T, 9)
        R_gt = etr(gts[:, :, :3]).view(B*T, 9)
        R_pred_proj = rpmg.simple_RPMG.apply(R_pred, 0.25, 0.01).view(B*T, 9)
        angle_loss = torch.nn.functional.l1_loss(R_pred_proj, R_gt)
        translation_loss = torch.nn.functional.l1_loss(poses[:, :, 3:], gts[:, :, 3:])
        return self.angle_weight * angle_loss + translation_loss

transform = new_custom_transform.Compose([
    new_custom_transform.ToTensor(),
    new_custom_transform.Resize((256, 512))
])

dataset = KITTI("kitti_data", train_seqs=['00', '01', '02', '04', '06', '09'],
                transform=transform, sequence_length=11)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

full_encoder = Encoder(params)
model = GNSSPoseNet(full_encoder).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = RPMGPoseLoss(angle_weight=100)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

checkpoint_dir = "E:/VITF/pretrained_models/gnss_stage2"
os.makedirs(checkpoint_dir, exist_ok=True)

start_epoch = 0
num_epochs = 10
best_val_loss = float('inf')
# resume_path = os.path.join(checkpoint_dir, "stage2_epoch_latest.pth")

# if os.path.exists(resume_path):
#     print(f"♻️ 恢复训练：加载 {resume_path}")
#     checkpoint = torch.load(resume_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch']
#     best_val_loss = checkpoint.get('best_val_loss', float('inf'))
# else:
print("🌟 从头开始训练 GNSS encoder + MLP head")

model.train()
for epoch in range(start_epoch, start_epoch + num_epochs):
    epoch_loss = 0.0
    for i, ((_, _, gnss, _, _), gts) in enumerate(tqdm(train_loader)):
        gnss, gts = gnss.to("cuda").float(), gts.to("cuda").float()
        optimizer.zero_grad()
        pred_pose = model(gnss)
        loss = criterion(pred_pose, gts)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / (i + 1)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for j, ((_, _, gnss, _, _), gts) in enumerate(val_loader):
            gnss, gts = gnss.to("cuda").float(), gts.to("cuda").float()
            pred_pose = model(gnss)
            loss = criterion(pred_pose, gts)
            val_loss += loss.item()
    avg_val_loss = val_loss / (j + 1)
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.6f}")
    model.train()

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_train_loss,
        'best_val_loss': best_val_loss
    }, os.path.join(checkpoint_dir, "stage2_epoch_latest.pth"))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.gnss_encoder.state_dict(), os.path.join(checkpoint_dir, "gnss_encoder_stage2_best.pth"))
        print("✅ 保存当前最佳模型: gnss_encoder_stage2_best.pth")

    scheduler.step()

# 最终保存 GNSS encoder 权重（带前缀）
prefixed_state_dict = {}
prefix = "Feature_net.gnss_encoder."
for k, v in model.gnss_encoder.state_dict().items():
    prefixed_state_dict[prefix + k] = v

torch.save(prefixed_state_dict, os.path.join(checkpoint_dir, "gnss_encoder_stage2_final.pth"))
print("✅ 最终模型保存完毕: gnss_encoder_stage2_final.pth")
