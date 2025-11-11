import sys
import os
import torch
import math
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

# ===== 路径导入 =====
sys.path.insert(0, '../')
from src.data.components.new_KITTI_dataset import KITTI
from src.utils import new_custom_transform
from src.models.components.new_vsvio import Encoder

# ===== 参数设置 =====
class ObjFromDict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

params = {
    "img_w": 512,
    "img_h": 256,
    "v_f_len": 512,
    "i_f_len": 256,
    "g_f_len": 128,
    "imu_dropout": 0.1,
    "seq_len": 11
}
params = ObjFromDict(params)

# ===== GNSS-only 模型结构 =====
class GNSSPoseNet(nn.Module):
    def __init__(self, encoder: Encoder, pose_dim=6):
        super().__init__()
        self.gnss_encoder = encoder.gnss_encoder  # 使用现有的 GNSS encoder 分支
        self.pose_head = nn.Sequential(
            nn.Linear(params.g_f_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pose_dim)
        )

    def forward(self, gnss):  # gnss: (B, T, 3)
        feat_g = self.gnss_encoder(gnss)  # (B, T, 128)
        return self.pose_head(feat_g)     # (B, T, 6)

# ===== 数据加载 =====
transform_train = new_custom_transform.Compose([
    new_custom_transform.ToTensor(),
    new_custom_transform.Resize((256, 512))
])

dataset = KITTI(
    "kitti_data",  # 请替换成你的KITTI数据根目录
    train_seqs=['00', '01', '02', '04', '06', '09'],
    transform=transform_train,
    sequence_length=11
)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# ===== 初始化模型 =====
full_encoder = Encoder(params)
model = GNSSPoseNet(full_encoder).to("cuda")

# ===== 优化器 & 损失函数 =====
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()

# ===== 训练设置 =====
checkpoint_dir = "E:/VITF/pretrained_models/gnss_stage1"
os.makedirs(checkpoint_dir, exist_ok=True)

start_epoch = 43
num_epochs = 27

# ===== 注释断点恢复部分 =====
resume_path = os.path.join(checkpoint_dir, "stage1_epoch_latest.pth")
if os.path.exists(resume_path):
    print(f"🔄 恢复训练：加载 {resume_path}")
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
else:
    print("🆕 从头开始训练 GNSS encoder + MLP head")

# ===== 正式训练 =====
model.train()
for epoch in range(start_epoch, start_epoch + num_epochs):
    epoch_loss = 0.0

    for i, ((imgs, imus, gnss, rot, w), gts) in enumerate(tqdm(loader)):
        gnss = gnss.to("cuda").float()  # (B, T, 3)
        gts = gts.to("cuda").float()    # (B, T, 6)

        optimizer.zero_grad()
        pred_pose = model(gnss)         # (B, T, 6)
        loss = criterion(pred_pose, gts)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / (i + 1)
    print(f"[Stage 1][Epoch {epoch+1}] Avg Loss: {avg_loss:.6f}")

    # ===== 保存当前模型（断点续训）=====
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, os.path.join(checkpoint_dir, "stage1_epoch_latest.pth"))

# ===== 最终保存 GNSS encoder 权重（带前缀）=====
gnss_state_dict = model.gnss_encoder.state_dict()
prefixed_state_dict = {}
prefix = "Feature_net.gnss_encoder."
for k, v in gnss_state_dict.items():
    new_key = prefix + k
    prefixed_state_dict[new_key] = v

torch.save(prefixed_state_dict, os.path.join(checkpoint_dir, "gnss_encoder_stage1_final.pth"))
print("✅ 训练完成，带前缀的 GNSS encoder 权重已保存：gnss_encoder_stage1_final.pth")


