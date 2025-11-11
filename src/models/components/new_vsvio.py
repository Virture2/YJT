import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
import numpy as np
import math

# class GNSS_encoder(nn.Module):
#     def __init__(self, opt):
#         super(GNSS_encoder, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=3,          # 每个时间步输入3维（如lat, lon, alt）
#             hidden_size=128,       # LSTM隐藏状态的维度是128
#             num_layers=2,          # 堆叠两层LSTM（更深，更复杂）
#             batch_first=True       # 输入tensor格式为(B, T, input_size)
#         )
#         self.project = nn.Linear(128, opt.g_f_len)
#     def forward(self, x):
#         # x: (B, T, 3)
#         out, _ = self.lstm(x)             # (B, T, 128)
#         out = self.project(out)           # (B, T, g_f_len)
#         return out

# class GNSS_encoder(nn.Module):
#     def __init__(self, opt):
#         super(GNSS_encoder, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=3,          # 每个时间步输入3维（如lat, lon, alt）
#             hidden_size=128,       # LSTM隐藏状态的维度是128
#             num_layers=2,          # 堆叠两层LSTM
#             batch_first=True,      # 输入tensor格式为(B, T, input_size)
#             bidirectional=True     # 双向LSTM
#         )
#         self.project = nn.Linear(128 * 2, opt.g_f_len)  # 因为是双向，输出维度是hidden_size*2
#
#     def forward(self, x):
#         # x: (B, T, 3)
#         out, _ = self.lstm(x)               # (B, T, 256)
#         out = self.project(out)             # (B, T, g_f_len)
#         return out

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
        # x 形状为 (B, T, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))  # (B*T, 11, 6)

        # 使用卷积层进行特征提取
        x = self.encoder_conv(x.permute(0, 2, 1))  # 转换为 (B*T, 256, 11)

        # 将卷积结果展平
        out = self.proj(x.reshape(x.shape[0], -1))  # 展平为 (B*T, i_f_len)

        # 还原回 (B, T, i_f_len)
        out = out.view(batch_size, seq_len, -1)

        return out

import torch.nn as nn

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
        # 此时 x 已经在 device 上，无需再 .to(self.device)
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

import torch
import torch.nn as nn
import math

# class GNSS_encoder(nn.Module):
#     def __init__(self,opt):
#         super(GNSS_encoder, self).__init__()
#
#         # 数值固定
#         input_dim = 3        # GNSS输入维度：x,y,z
#         embed_dim = 128      # embedding维度
#         num_layers = 2       # Transformer层数
#         nhead = 8            # 多头注意力头数
#         dim_feedforward = 512
#         dropout = 0.1
#
#         self.embedding = nn.Linear(input_dim, embed_dim)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.project = nn.Linear(embed_dim, opt.g_f_len)
#         self.embed_dim = embed_dim
#
#     def positional_embedding(self, seq_length, device):
#         pos = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=device).float() * -(math.log(10000.0)/self.embed_dim))
#         pe = torch.zeros(seq_length, self.embed_dim, device=device)
#         pe[:, 0::2] = torch.sin(pos * div_term)
#         pe[:, 1::2] = torch.cos(pos * div_term)
#         return pe.unsqueeze(0)  # shape: (1, T, D)
#
#     def forward(self, x):
#         # x: (B, T, 3)
#         B, T, _ = x.size()
#         device = x.device
#         x = self.embedding(x)                           # (B, T, 128)
#         x = x + self.positional_embedding(T, device)   # 加位置编码
#         x = self.transformer(x)                        # Transformer编码
#         x = self.project(x)                            # 输出 (B, T, 128)
#         return x

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys
import os

# 将脚本所在目录加入 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from extractor import ResNetFPN
#
# class Encoder(nn.Module):
#     def __init__(self, opt, raft_args, device='cuda'):
#         super().__init__()
#         self.device = device
#
#         # RAFT 视觉特征提取器
#         self.raft = RAFT(raft_args).to(self.device)
#
#         # 通过 dummy 图像计算 RAFT 输出通道
#         dummy_img = torch.zeros(1, 3, opt.img_h, opt.img_w).to(self.device)
#         with torch.no_grad():
#             raft_out = self.raft.fnet(dummy_img)  # (1, C, H/8, W/8)
#         feat_dim = raft_out.shape[1]
#
#         # 将池化后的特征映射到 512 维
#         self.visual_head = nn.Linear(feat_dim, 512).to(self.device)
#
#         # IMU 编码器
#         self.inertial_encoder = Inertial_encoder(opt, device=self.device)
#
#         # GNSS 编码器
#         self.gnss_encoder = GNSS_encoder(opt).to(self.device)
#
#     def forward(self, img, imu, gnss):
#         B, seq_len, C, H, W = img.shape
#
#         # 序列两两帧展开
#         img1_batch = img[:, :-1].reshape(-1, 3, H, W)  # (B*(seq_len-1), 3, H, W)
#         img2_batch = img[:, 1:].reshape(-1, 3, H, W)
#
#         with torch.no_grad():  # RAFT 不训练
#             fmap1 = self.raft.fnet(img1_batch)  # (B*(seq_len-1), C, H', W')
#             fmap2 = self.raft.fnet(img2_batch)
#
#             # 计算相关性并沿 H,W 维度池化
#             corr_map = fmap1 * fmap2           # (B*(seq_len-1), C, H', W')
#             pooled = corr_map.mean(dim=[2,3])  # (B*(seq_len-1), C)
#
#         # 映射到 512 维
#         v = self.visual_head(pooled)           # (B*(seq_len-1), 512)
#         v = v.view(B, seq_len-1, 512)          # (B, seq_len-1, 512)
#
#         # IMU 特征
#         imu_chunks = [imu[:, i*10:i*10+11, :].unsqueeze(1).to(self.device) for i in range(seq_len-1)]
#         imu = torch.cat(imu_chunks, dim=1)
#         imu = self.inertial_encoder(imu)
#
#         # GNSS 特征
#         g = self.gnss_encoder(gnss)
#
#         return v, imu, g
#
#
#
# # RAFT
# from raft import RAFT
#
# class Encoder(nn.Module):
#     def __init__(self, opt, raft_args, device='cuda'):
#         super().__init__()
#         self.device = device
#
#         # RAFT 视觉特征提取器
#         self.raft = RAFT(raft_args).to(self.device)
#
#         # 通过 dummy 图像计算 RAFT 输出通道
#         dummy_img = torch.zeros(1, 3, opt.img_h, opt.img_w).to(self.device)
#         raft_out = self.raft(dummy_img, dummy_img, test_mode=True)['final']  # (1, 2, H/8, W/8)
#         pooled = raft_out.mean(dim=[2,3])  # (1, 2)
#         feat_dim = pooled.shape[1]  # 这里是 2
#
#         # 将池化后的特征映射到 512 维
#         self.visual_head = nn.Linear(feat_dim, 512).to(self.device)
#
#         # IMU 编码器
#         self.inertial_encoder = Inertial_encoder(opt, device=self.device)
#
#         # GNSS 编码器
#         self.gnss_encoder = GNSS_encoder(opt).to(self.device)
#
#     def forward(self, img, imu, gnss):
#         B, seq_len, C, H, W = img.shape
#
#         # --------------------------------------------
#         # 1️⃣ 构造相邻帧对一次性 batch
#         # img1_batch: (B*(seq_len-1), 3, H, W)
#         # img2_batch: (B*(seq_len-1), 3, H, W)
#         img1_batch = img[:, :-1].reshape(-1, C, H, W)
#         img2_batch = img[:, 1:].reshape(-1, C, H, W)
#
#         # --------------------------------------------
#         # 2️⃣ RAFT 前向
#         with torch.no_grad():  # RAFT 不训练时可以加
#             raft_out = self.raft(img1_batch, img2_batch, test_mode=True)['final']  # (B*(seq_len-1), 2, H/8, W/8)
#
#         # --------------------------------------------
#         # 3️⃣ 池化 + 映射到 512 维
#         pooled = raft_out.mean(dim=[2,3])  # (B*(seq_len-1), 2)
#         v = self.visual_head(pooled).view(B, seq_len-1, 512)  # (B, seq_len-1, 512)
#
#         # --------------------------------------------
#         # 4️⃣ IMU 特征
#         imu_chunks = [imu[:, i*10:i*10+11, :].unsqueeze(1).to(self.device) for i in range(seq_len-1)]
#         imu = torch.cat(imu_chunks, dim=1)
#         imu = self.inertial_encoder(imu)
#
#         # --------------------------------------------
#         # 5️⃣ GNSS 特征
#         g = self.gnss_encoder(gnss)
#
#         return v, imu, g


# # 修改 Encoder 类，接受 device 参数并使用它
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
        # ========================
        # Visual Feature: 拼接相邻图像 (10 对相邻帧)
        # ========================
        # print(f"Input img device2: {img.device}")
        # print(f"Input imu device2: {imu.device}")
        # print(f"Input gnss device2: {gnss.device}")

        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2).to(self.device)  # (B, 10, 6, H, W)
        batch_size, seq_len = v.size(0), v.size(1)
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))  # (B*10, 6, H, W)
        v = self.encode_image(v)  # e.g. (B*10, C, H', W')
        v = v.view(batch_size, seq_len, -1)  # (B, 10, flattened)
        v = self.visual_head(v)  # (B, 10, v_f_len)

        # print(f"Visual feature device: {v.device}")
        # ========================
        # IMU Feature: 10 段，每段11条数据
        # ========================
        imu_chunks = [imu[:, i * 10:i * 10 + 11, :].unsqueeze(1).to(self.device) for i in range(seq_len)]  # 加入到设备上
        imu = torch.cat(imu_chunks, dim=1).to(self.device)  # (B, 10, 11, 6)
        imu = self.inertial_encoder(imu)  # (B, 10, i_f_len)

        # print(f"Inertial feature device: {imu.device}")
        # ========================
        # GNSS Feature: 每步一个差分，shape (B, 10, 3)
        # ========================
        g = self.gnss_encoder(gnss)  # 加入到设备上
        # print(f"GNSS feature device: {g.device}")
        return v, imu, g

class Fusion_module(nn.Module):
    def __init__(self, opt):
        super(Fusion_module, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.v_f_len + opt.i_f_len + opt.g_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(nn.Linear(self.f_len, 2 * self.f_len))

    def forward(self, v, i, g):
        feat_cat = torch.cat((v, i, g), -1)
        if self.fuse_method == 'cat':
            return feat_cat
        elif self.fuse_method == 'soft':
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[..., 0]


class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()
        self.fuse = Fusion_module(opt)
        self.rnn = nn.LSTM(
            input_size=self.fuse.f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True
        )
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6)
        )

    def forward(self, fv, fi, fg, dec=None, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())

        fused = self.fuse(fv, fi, fg)
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc


class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()
        self.Feature_net = Encoder(opt)
        self.Pose_net = Pose_RNN(opt)
        initialization(self)

    def forward(self, img, imu, gnss, hc=None):
        fv, fi, fg = self.Feature_net(img, imu, gnss)
        pose, hc = self.Pose_net(fv, fi, fg, dec=None, prev=hc)
        return pose, hc


def initialization(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    kaiming_normal_(param)
                elif 'bias' in name:
                    param.data.zero_()
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
