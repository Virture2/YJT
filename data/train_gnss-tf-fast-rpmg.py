import sys
import os
import math
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from path import Path
import torch.nn as nn
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# ====== 导入工具函数 ======
from src.utils.kitti_utils import eulerAnglesToRotationMatrixTorch as etr
from src.utils import rpmg

# ====== 自定义损失函数 ======
class RPMGPoseLoss(nn.Module):
    def __init__(self, angle_weight):
        super().__init__()
        self.angle_weight = angle_weight

    def forward(self, poses, gts, weights=None, use_weighted_loss=True):
        angle_loss = torch.nn.functional.l1_loss(
            rpmg.simple_RPMG.apply(
                etr(poses[:, :, :3]).view(poses.shape[0]*poses.shape[1], 9),
                1/4,
                0.01
            ).view(-1, 9),
            etr(gts[:, :, :3]).view(poses.shape[0]*poses.shape[1], 9)
        )
        translation_loss = torch.nn.functional.l1_loss(poses[:, :, 3:], gts[:, :, 3:])
        pose_loss = self.angle_weight * angle_loss + translation_loss
        return pose_loss

# ====== GNSS 读取函数 ======
def read_gnss_from_text(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            data.append([x, y, z])
    return np.array(data)

# ====== Dataset ======
class LatentVectorDataset(Dataset):
    def __init__(self, root, root_dir,train_seqs, sequence_length=11):
        self.root = Path(root)
        self.root_dir = Path(root_dir)
        print(self.root_dir)
        self.sequence_length = sequence_length
        self.train_seqs = train_seqs
        print(self.train_seqs)
        self.make_dataset()

    def make_dataset(self):
        sequence_set = []
        for folder in self.train_seqs:
            gnss = read_gnss_from_text(self.root / 'oxts' / folder / f'{folder}.txt')
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

        latent_vector = torch.from_numpy(latent_vector).float()
        gnss = torch.from_numpy(gnss).float()
        rot = torch.from_numpy(rot).float()
        w = torch.from_numpy(w).float()
        gt = torch.from_numpy(gt).float()

        if gt.dim() == 4 and gt.size(1) == 1:
            gt = gt.squeeze(1)
        elif gt.dim() == 3 and gt.size(0) == 1:
            gt = gt.squeeze(0)

        return (latent_vector, gnss, rot, w), gt

    def __len__(self):
        return len(self.samples)

# ====== 模型组件 ======
class DoubleLinearQKV(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.k = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.v = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)

class CustomSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, q, k, v):
        B, T, _ = q.size()
        def split_heads(x): return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(attn_output)

class GNSS_encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.embedding_dim = opt.g_f_len
        self.embedding = nn.Linear(3, self.embedding_dim)
        self.qkv = DoubleLinearQKV(self.embedding_dim, 128, self.embedding_dim)
        self.attention = CustomSelfAttention(self.embedding_dim, 8)
        self.ffn = nn.Sequential(nn.Linear(self.embedding_dim, 128), nn.ReLU(), nn.Linear(128, self.embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8, dim_feedforward=512,
                                                   dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.project = nn.Linear(self.embedding_dim, opt.g_f_len)

    def positional_embedding(self, seq_length, device):
        pos = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=device).float() * -(math.log(10000.0)/self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim, device=device)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)

    def forward(self, x):
        B, T, _ = x.size()
        device = x.device
        x = self.embedding(x)
        pos_embedding = self.positional_embedding(T, device)
        x = x + pos_embedding
        q, k, v = self.qkv(x)
        attn_out = self.attention(q, k, v)
        x = x + attn_out
        x = x + self.ffn(x)
        x = self.transformer_encoder(x)
        return self.project(x)

class PoseTransformer(nn.Module):
    def __init__(self, input_dim=896, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1, device='cuda'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.fc1 = nn.Linear(input_dim, self.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc2 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.Linear(self.embedding_dim, 6))
        self.to(self.device)

    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=self.device).float() * -(math.log(10000.0)/self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim, device=self.device)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf"), device=self.device), diagonal=1)

    def forward(self, batch):
        device = next(self.fc1.parameters()).device
        visual_inertial_features = batch.to(device)
        seq_length = visual_inertial_features.size(1)
        pos_embedding = self.positional_embedding(seq_length).to(device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding
        mask = self.generate_square_subsequent_mask(seq_length).to(device)
        output = self.transformer_encoder(visual_inertial_features, mask=mask)
        return self.fc2(output)

class FeatureEncodingModel(nn.Module):
    def __init__(self, params, device='cuda'):
        super().__init__()
        self.device = device
        self.gnss_encoder = GNSS_encoder(params)
        self.pose_transformer = PoseTransformer(input_dim=params.v_f_len + params.i_f_len + params.g_f_len, device=self.device)
        self.to(self.device)

    def forward(self, LatentVector, gnss):
        gnss = gnss.to(self.device)
        feat_g = self.gnss_encoder(gnss)
        feat_all = torch.cat([LatentVector, feat_g], dim=-1)
        return self.pose_transformer(feat_all)

# ====== 参数设置 ======
class ObjFromDict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

params = ObjFromDict({
    "img_w": 512, "img_h": 256, "v_f_len": 512, "i_f_len": 256, "g_f_len": 128, "imu_dropout": 0.1, "seq_len": 11
})

# ====== 数据集和模型 ======
train_dataset = LatentVectorDataset("kitti_data", root_dir="kitti_latent_data/vift",
                                    train_seqs=['00', '01', '02', '04', '06', '09'], sequence_length=11)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)

val_dataset = LatentVectorDataset("kitti_data", root_dir="kitti_latent_data/vift_val",
                                  train_seqs=['05', '07', '10'], sequence_length=11)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

device = 'cuda:0'
model = FeatureEncodingModel(params, device=device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = RPMGPoseLoss(angle_weight=40)

checkpoint_dir = "E:/VITF/pretrained_models/fast_transformer_rpmg_process"
os.makedirs(checkpoint_dir, exist_ok=True)

best_val_loss = float('inf')
best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

# ====== 训练 ======
model.train()
for epoch in range(50):
    epoch_loss = 0.0
    for i, ((LatentVector, gnss, rot, w), gts) in enumerate(tqdm(train_loader)):
        LatentVector, gnss, gts = LatentVector.to(device).float(), gnss.to(device).float(), gts.to(device).float()
        optimizer.zero_grad()
        pred_pose = model(LatentVector, gnss)
        loss = criterion(pred_pose, gts)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / (i + 1)
    print(f"[Epoch {epoch + 1}] Train Avg Loss: {avg_loss:.6f}")

    # ====== 验证 ======
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for j, ((LatentVector, gnss, rot, w), gts) in enumerate(val_loader):
            LatentVector, gnss, gts = LatentVector.to(device).float(), gnss.to(device).float(), gts.to(device).float()
            pred_pose = model(LatentVector, gnss)
            loss = criterion(pred_pose, gts)
            val_loss_total += loss.item()
    val_loss_avg = val_loss_total / (j + 1)
    print(f"[Epoch {epoch + 1}] Val Avg Loss: {val_loss_avg:.6f}")

    # ====== 保存每轮权重 ======
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_loss,
        'val_loss': val_loss_avg
    }, os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth"))

    # ====== 更新最佳模型 ======
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model updated at epoch {epoch + 1}, val_loss: {best_val_loss:.6f}")

    model.train()

# ====== 保存最终模型 ======
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "gnss_encoder_transformer_final.pth"))
