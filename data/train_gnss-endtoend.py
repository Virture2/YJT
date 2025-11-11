import sys
import os
import torch
import math
from tqdm import tqdm
import torch.nn as nn
sys.path.insert(0, '../')
from src.data.components.new_KITTI_dataset import KITTI
from src.utils import new_custom_transform
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# from src.models.components.new_vsvio import Encoder

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0.0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

class Inertial_encoder(nn.Module):
    def __init__(self, opt, device='cuda'):
        super(Inertial_encoder, self).__init__()
        self.device = device  # 设备信息

        # 使用 1D 卷积层进行特征提取
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),  # 第一层卷积
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout)
        ).to(self.device)  # 确保整个序列都在指定的设备上

        # 投影层，将卷积后的输出映射到最终特征空间
        self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len).to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))  # (B*T, 11, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))  # 转换为 (B*T, 256, 11)
        out = self.proj(x.reshape(x.shape[0], -1))  # 展平为 (B*T, i_f_len)
        out = out.view(batch_size, seq_len, -1)
        return out

class DoubleLinearQKV(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.k = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v

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

        def split_heads(x):
            return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, head_dim]

        # 不再调用 .to(self.device) —— 输入已处理好
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(attn_output)

class GNSS_encoder(nn.Module):
    def __init__(self, opt):
        super(GNSS_encoder, self).__init__()

        self.embedding_dim = opt.g_f_len  # 输入和输出的维度
        self.embedding = nn.Linear(3, self.embedding_dim)
        self.qkv = DoubleLinearQKV(self.embedding_dim, 128, self.embedding_dim)
        self.attention = CustomSelfAttention(self.embedding_dim, 8)
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True  # 保证不再 permute
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.project = nn.Linear(self.embedding_dim, opt.g_f_len)
    def positional_embedding(self, seq_length, device):
        pos = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=device).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim, device=device)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)  # shape: (1, T, D)
    def forward(self, x):
        # x: (B, T, 3)
        B, T, _ = x.size()
        device = x.device
        x = self.embedding(x)  # (B, T, D)
        pos_embedding = self.positional_embedding(T, device)
        x = x + pos_embedding
        q, k, v = self.qkv(x)
        attn_out = self.attention(q, k, v)
        x = x + attn_out
        x = x + self.ffn(x)
        x = self.transformer_encoder(x)  # batch_first=True
        x = self.project(x)
        return x


class Encoder(nn.Module):
    def __init__(self, opt, device='cuda'):
        super(Encoder, self).__init__()
        self.device = device  # 保存设备信息

        # Visual Encoder
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2).to(self.device)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2).to(self.device)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2).to(self.device)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2).to(self.device)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2).to(self.device)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2).to(self.device)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2).to(self.device)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2).to(self.device)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5).to(self.device)
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h)).to(self.device)  # 转移到指定设备
        __tmp = self.encode_image(__tmp)
        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len).to(self.device)
        # Inertial Encoder
        self.inertial_encoder = Inertial_encoder(opt, device=self.device)
        # GNSS Encoder
        self.gnss_encoder = GNSS_encoder(opt).to(self.device)

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def forward(self, img, imu, gnss):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2).to(self.device)  # (B, 10, 6, H, W)
        batch_size, seq_len = v.size(0), v.size(1)
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))  # (B*10, 6, H, W)
        v = self.encode_image(v)  # e.g. (B*10, C, H', W')
        v = v.view(batch_size, seq_len, -1)  # (B, 10, flattened)
        v = self.visual_head(v)  # (B, 10, v_f_len)
        imu_chunks = [imu[:, i * 10:i * 10 + 11, :].unsqueeze(1).to(self.device) for i in range(seq_len)]  # 加入到设备上
        imu = torch.cat(imu_chunks, dim=1).to(self.device)  # (B, 10, 11, 6)
        imu = self.inertial_encoder(imu)  # (B, 10, i_f_len)
        g = self.gnss_encoder(gnss)  # 加入到设备上
        return v, imu, g

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
        self.Feature_net = Encoder(params, device=self.device)
        self.pose_transformer = PoseTransformer(input_dim=params.v_f_len + params.i_f_len + params.g_f_len,
                                                device=self.device)
        self.to(self.device)

    def forward(self, imgs, imus, gnss):
        imgs = imgs.to(self.device)
        imus = imus.to(self.device)
        gnss = gnss.to(self.device)

        feat_v, feat_i, feat_g = self.Feature_net(imgs, imus, gnss)

        feat_all = torch.cat([feat_v, feat_i, feat_g], dim=-1).to(self.device)

        output = self.pose_transformer(feat_all)

        return feat_v, feat_i, feat_g, output

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

# ===== 数据集 =====
transform_train = new_custom_transform.Compose([
    new_custom_transform.ToTensor(),
    new_custom_transform.Resize((256, 512))
])

dataset = KITTI(
    "kitti_data",
    train_seqs=['00', '01', '02', '04', '06', '09'],
    transform=transform_train,
    sequence_length=11
)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# ===== 实例化模型 =====
device = 'cuda:0'
model = FeatureEncodingModel(params, device=device)

# ===== 加载预训练视觉+IMU权重 =====
pretrained_w = torch.load("../pretrained_models/vf_512_if_256_3e-05.model", map_location=device)
# print(f"Pretrained weights loaded on device: {next(iter(pretrained_w.values())).device}")
model_dict = model.state_dict()
update_dict = {k: v for k, v in pretrained_w.items() if
               k in model_dict and "gnss_encoder" not in k and "pose_transformer" not in k}
print(f"✔️ 加载了 {len(update_dict)} 个视觉/IMU权重")
model_dict.update(update_dict)
model.load_state_dict(model_dict)

# GNSS编码器权重
tot_dict_gnss = {k: v for k, v in model_dict.items() if "gnss_encoder" in k}
print(f"GNSS编码器权重数量: {len(tot_dict_gnss)}")

# PoseTransformer权重
tot_dict_pose = {k: v for k, v in model_dict.items() if "pose_transformer" in k}
print(f"PoseTransformer权重数量: {len(tot_dict_pose)}")

# ===== 冻结视觉+IMU部分参数 =====
def freeze_non_gnss(model):
    for name, param in model.named_parameters():
        if "gnss_encoder" not in name and "pose_transformer" not in name:
            param.requires_grad = False
            # print(f"❄️ 冻结：{name}")


freeze_non_gnss(model)

# ===== 优化器和损失函数 =====
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# criterion = nn.L1Loss()
criterion = RPMGPoseLoss(angle_weight=40)


# ===== 训练设置 =====
checkpoint_dir = "E:/VITF/pretrained_models/transformer_process_rpmg"
os.makedirs(checkpoint_dir, exist_ok=True)

resume_path = os.path.join(checkpoint_dir, "model_epoch_50.pth")
start_epoch = 50

if os.path.exists(resume_path):
    print(f"🔄 恢复训练：加载 {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  # 从第10轮后继续
else:
    print("⚠️ 未找到上次模型，默认从头开始训练")

# ===== 正式训练 =====
model.train()
for epoch in range(start_epoch, start_epoch + 200):  # 再训练40轮
    epoch_loss = 0.0
    print(start_epoch)
    print(epoch)
    for i, ((imgs, imus, gnss, rot, w), gts) in enumerate(tqdm(loader)):
        imgs = imgs.to(device).float()
        imus = imus.to(device).float()
        gnss = gnss.to(device).float()
        gts = gts.to(device).float()

        optimizer.zero_grad()
        _, _, _, pred_pose = model(imgs, imus, gnss)
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
