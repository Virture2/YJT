import sys
import os
import math
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from path import Path

sys.path.insert(0, '../')
# from src.data.components.new_KITTI_dataset import KITTI
import torch.nn as nn
import numpy as np
# from src.models.components.new_vsvio import Encoder

def read_gnss_from_text(file_path):
    """
    从GPS ENU坐标文件加载数据
    返回: Nx3 numpy数组
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            data.append([x, y, z])
    return np.array(data)


class LatentVectorDataset(Dataset):
    def __init__(self, root,root_dir,
                 sequence_length=11,
                 train_seqs=['00', '01', '02', '04', '06', '09'],
                 transform=None):

        self.root = Path(root)
        self.root_dir = Path(root_dir)
        # print(f"self.root path: {self.root}")
        # print(f"self.root_dir path: {self.root_dir}")
        self.sequence_length = sequence_length
        self.train_seqs = train_seqs
        self.make_dataset()

    def make_dataset(self):
        sequence_set = []
        for folder in self.train_seqs:
            gnss = read_gnss_from_text(self.root / 'oxts' / folder / '{}.txt'.format(folder))
            # print(f"Sequence {folder}: GNSS length = {len(gnss)}")
            for i in range(len(gnss) - self.sequence_length):
                gnss_samples = gnss[i + 1:i + self.sequence_length] - gnss[i:i + self.sequence_length - 1]
                sample = {'gnss': gnss_samples}
                sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, idx):
        latent_vector = np.load(os.path.join(self.root_dir, f"{idx}.npy"))
        gt = np.load(os.path.join(self.root_dir, f"{idx}_gt.npy"))
        rot = np.load(os.path.join(self.root_dir, f"{idx}_rot.npy"))
        w = np.load(os.path.join(self.root_dir, f"{idx}_w.npy"))
        gnss = np.array(self.samples[idx]['gnss'])

        # 转成 torch tensor 并保证 dtype
        latent_vector = torch.from_numpy(latent_vector).float()
        gnss = torch.from_numpy(gnss).float()
        rot = torch.from_numpy(rot).float()
        w = torch.from_numpy(w).float()

        # GT shape squeeze，确保 (T, 6)
        gt = torch.from_numpy(gt).float()
        if gt.dim() == 4 and gt.size(1) == 1:  # 去掉多余的维度
            gt = gt.squeeze(1)  # [B, 1, T, 6] -> [B, T, 6]
        elif gt.dim() == 3 and gt.size(0) == 1:
            gt = gt.squeeze(0)

        return (latent_vector, gnss, rot, w), gt

    def __len__(self):
        return len(self.samples)


class GNSS_encoder(nn.Module):
    def __init__(self,opt):
        super(GNSS_encoder, self).__init__()

        # 数值固定
        input_dim = 3        # GNSS输入维度：x,y,z
        embed_dim = 128      # embedding维度
        num_layers = 2       # Transformer层数
        nhead = 8            # 多头注意力头数
        dim_feedforward = 512
        dropout = 0.1

        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.project = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def positional_embedding(self, seq_length, device):
        pos = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=device).float() * -(math.log(10000.0)/self.embed_dim))
        pe = torch.zeros(seq_length, self.embed_dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)  # shape: (1, T, D)

    def forward(self, x):
        # x: (B, T, 3)
        B, T, _ = x.size()
        device = x.device
        x = self.embedding(x)                           # (B, T, 128)
        x = x + self.positional_embedding(T, device)   # 加位置编码
        x = self.transformer(x)                        # Transformer编码
        x = self.project(x)                            # 输出 (B, T, 128)
        return x


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

class PoseTransformer(nn.Module):
    def __init__(self, input_dim=896, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1,
                 device='cuda'):
        super(PoseTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6)
        )
        self.to(self.device)

    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=self.device).float() * -(
                        math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim, device=self.device)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(
            torch.full((sz, sz), float("-inf"), dtype=torch.float32, device=self.device),
            diagonal=1
        )

    def forward(self, batch):
        device = next(self.fc1.parameters()).device
        visual_inertial_features = batch.to(device)
        seq_length = visual_inertial_features.size(1)
        pos_embedding = self.positional_embedding(seq_length).to(device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding
        mask = self.generate_square_subsequent_mask(seq_length).to(device)
        output = self.transformer_encoder(visual_inertial_features, mask=mask)
        output = self.fc2(output)
        return output


class FeatureEncodingModel(nn.Module):
    def __init__(self, params, device='cuda'):
        super().__init__()
        self.device = device
        self.gnss_encoder = GNSS_encoder(params)
        self.pose_transformer = PoseTransformer(input_dim=params.v_f_len + params.i_f_len + params.g_f_len,
                                                device=self.device)
        self.to(self.device)

    def forward(self, LatentVector, gnss):
        gnss = gnss.to(self.device)

        feat_g = self.gnss_encoder(gnss)

        # print(f"LatentVector shape: {LatentVector.shape}")
        # print(f"feat_g shape: {feat_g.shape}")
        feat_all = torch.cat([LatentVector, feat_g], dim=-1)
        # print(f"feat_all shape: {feat_all.shape}")

        output = self.pose_transformer(feat_all)

        return output

# ===== 数据集 =====

dataset = LatentVectorDataset(
    "kitti_data",
    root_dir="kitti_latent_data/vift",
    train_seqs=['00', '01', '02', '04', '06', '09'],
    sequence_length=11
)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# ===== 实例化模型 =====
device = 'cuda:0'
model = FeatureEncodingModel(params, device=device)

model_dict = model.state_dict()

# GNSS编码器权重
tot_dict_gnss = {k: v for k, v in model_dict.items() if "gnss_encoder" in k}
print(f"GNSS编码器权重数量: {len(tot_dict_gnss)}")

# PoseTransformer权重
tot_dict_pose = {k: v for k, v in model_dict.items() if "pose_transformer" in k}
print(f"PoseTransformer权重数量: {len(tot_dict_pose)}")


# ===== 优化器和损失函数 =====
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.L1Loss()

# ===== 训练设置 =====
checkpoint_dir = "E:/VITF/pretrained_models/fast_transformer_process_nortf"
os.makedirs(checkpoint_dir, exist_ok=True)

# resume_path = os.path.join(checkpoint_dir, "model_epoch_60.pth")
start_epoch = 0

# if os.path.exists(resume_path):
#     print(f"🔄 恢复训练：加载 {resume_path}")
#     checkpoint = torch.load(resume_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch']  # 从第10轮后继续
# else:
#     print("⚠️ 未找到上次模型，默认从头开始训练")

# ===== 正式训练 =====
model.train()
for epoch in range(start_epoch, start_epoch + 200):  # 再训练40轮
    epoch_loss = 0.0
    print(epoch)
    for i, ((LatentVector, gnss, rot, w), gts) in enumerate(tqdm(loader)):
        LatentVector = LatentVector.to(device).float()
        gnss = gnss.to(device).float()
        gts = gts.to(device).float()

        optimizer.zero_grad()
        pred_pose = model(LatentVector, gnss)
        # print(f"Batch {i} pred_pose shape: {pred_pose.shape}")

        loss = criterion(pred_pose, gts)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / (i + 1)
    print(f"[Epoch {epoch + 1}] Avg Loss: {avg_loss:.6f}")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth"))

# ===== 最终模型保存 =====
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "gnss_encoder_transformer_final.pth"))
