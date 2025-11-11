# import sys, os, torch, torch.nn as nn
# from tqdm import tqdm
#
# sys.path.insert(0, '../')
# from src.data.components.new_KITTI_dataset import KITTI
# from src.utils import new_custom_transform
# from src.models.components.new_vsvio import Encoder
#
# # ===== GNSS Only 模型 =====
# class GNSSOnlyModel(nn.Module):
#     def __init__(self, encoder, g_f_len):
#         super().__init__()
#         self.gnss_encoder = encoder.gnss_encoder
#         self.mlp_head = nn.Sequential(
#             nn.Linear(g_f_len, 128),
#             nn.LeakyReLU(0.1),
#             nn.Linear(128, 6)
#         )
#
#     def forward(self, gnss):
#         feat_g = self.gnss_encoder(gnss)  # (B, T, g_f_len)
#         return self.mlp_head(feat_g)      # (B, T, 6)
#
# # ===== 主函数入口 =====
# def main():
#     # --- 参数配置 ---
#     params = {
#         "img_w": 512, "img_h": 256,
#         "v_f_len": 512, "i_f_len": 256,
#         "g_f_len": 128, "imu_dropout": 0.1,
#         "seq_len": 11
#     }
#     class Obj: pass
#     opt = Obj(); [setattr(opt, k, v) for k, v in params.items()]
#
#     # --- 数据加载 ---
#     transform = new_custom_transform.Compose([
#         new_custom_transform.ToTensor(),
#         new_custom_transform.Resize((256, 512))
#     ])
#     dataset = KITTI("kitti_data", train_seqs=['00', '01', '02', '04', '06', '09'],
#                     transform=transform, sequence_length=11)
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=2,
#         shuffle=True
#     )
#
#     # --- 模型定义 ---
#     encoder = Encoder(opt)
#     model = GNSSOnlyModel(encoder, g_f_len=opt.g_f_len).cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     criterion = nn.L1Loss()
#
#     # --- 模型保存路径 ---
#     checkpoint_dir = "E:/VITF/pretrained_models/gnss_stage1"
#     os.makedirs(checkpoint_dir, exist_ok=True)
#
#     # --- 断点恢复 ---
#     start_epoch = 1
#     for ep in range(20, 0, -1):
#         resume_path = os.path.join(checkpoint_dir, f"gnss_stage1_epoch{ep}.pth")
#         if os.path.exists(resume_path):
#             model.load_state_dict(torch.load(resume_path))
#             start_epoch = ep + 1
#             print(f"🔄 恢复训练：从 epoch {start_epoch} 开始")
#             break
#
#     # --- 正式训练 ---
#     model.train()
#     for epoch in range(start_epoch, 21):
#         epoch_loss = 0
#         for (imgs, imus, gnss, _, _), gts in tqdm(loader, desc=f"Epoch {epoch}"):
#             gnss, gts = gnss.cuda().float(), gts.cuda().float()
#             optimizer.zero_grad()
#             pred = model(gnss)
#             loss = criterion(pred, gts)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#
#         avg_loss = epoch_loss / len(loader)
#         print(f"[Stage 1][Epoch {epoch}] Avg Loss: {avg_loss:.6f}")
#
#         # --- GNSS Encoder 参数统计 ---
#         total_param_count = 0
#         for name, param in model.named_parameters():
#             if "gnss_encoder" in name and param.requires_grad:
#                 param_count = param.numel()
#                 total_param_count += param_count
#                 print(f"🔧 {name:55s} | shape: {tuple(param.shape)} | numel: {param_count:,}")
#
#         with torch.no_grad():
#             gnss_weights = [p.view(-1) for n, p in model.named_parameters() if "gnss_encoder" in n and p.requires_grad]
#             all_weights = torch.cat(gnss_weights)
#             print(f"📊 GNSS Encoder Stats — Total Params: {total_param_count:,} | Mean: {all_weights.mean().item():.6f} | Std: {all_weights.std().item():.6f}")
#
#         # --- 保存模型权重 ---
#         save_path = os.path.join(checkpoint_dir, f"gnss_stage1_epoch{epoch}.pth")
#         torch.save(model.state_dict(), save_path)
#         print(f"✅ 模型已保存: {save_path}")
#
# # ===== Windows 多进程安全入口 =====
# if __name__ == "__main__":
#     main()


# train_stage2_finetune_transformer.py
import sys, os, torch
import torch.nn as nn
from tqdm import tqdm
import math

# === 项目路径配置 ===
sys.path.insert(0, "E:/VITF")  # ✅ 修改为你项目根目录

from src.data.components.new_KITTI_dataset import KITTI
from src.utils import new_custom_transform
from src.models.components.new_vsvio import Encoder

# ===== Positional Transformer Head =====
class PoseTransformer(nn.Module):
    def __init__(self, input_dim=896, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                       dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.fc2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_dim, 6)
        )

    def positional_embedding(self, seq_length, device):
        pos = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=device).float() * -(math.log(10000.0) / self.embedding_dim))
        pe = torch.zeros(seq_length, self.embedding_dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        B, T, _ = x.size()
        x = self.fc1(x)
        x = x + self.positional_embedding(T, x.device)
        x = self.encoder(x)
        return self.fc2(x)

# ===== GNSS Only + Transformer 模型（dummy v/i）=====
class GNSSWithTransformer(nn.Module):
    def __init__(self, encoder, g_f_len=128, total_f_len=896):
        super().__init__()
        self.gnss_encoder = encoder.gnss_encoder
        self.transformer = PoseTransformer(input_dim=total_f_len)

    def forward(self, gnss):
        feat_g = self.gnss_encoder(gnss)
        B, T, _ = feat_g.shape
        dummy_v = torch.zeros(B, T, 512).to(feat_g.device)
        dummy_i = torch.zeros(B, T, 256).to(feat_g.device)
        feat_all = torch.cat([dummy_v, dummy_i, feat_g], dim=-1)  # shape (B, T, 896)
        return self.transformer(feat_all)

# ===== 训练主函数 =====
def main():
    # --- 配置参数 ---
    params = {
        "img_w": 512, "img_h": 256,
        "v_f_len": 512, "i_f_len": 256,
        "g_f_len": 128, "imu_dropout": 0.1,
        "seq_len": 11
    }
    class Obj: pass
    opt = Obj(); [setattr(opt, k, v) for k, v in params.items()]

    # --- 数据加载 ---
    transform = new_custom_transform.Compose([
        new_custom_transform.ToTensor(),
        new_custom_transform.Resize((256, 512))
    ])
    dataset = KITTI("kitti_data", train_seqs=['00', '01', '02', '04', '06', '09'],
                    transform=transform, sequence_length=11)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # --- 模型定义 ---
    encoder = Encoder(opt)
    model = GNSSWithTransformer(encoder).cuda()

    # --- 加载 Stage1 的 GNSS encoder 权重 ---
    gnss_stage1_path = "E:/VITF/pretrained_models/gnss_stage1/gnss_stage1_epoch20.pth"
    if os.path.exists(gnss_stage1_path):
        gnss_weights = torch.load(gnss_stage1_path, map_location="cpu")
        model.gnss_encoder.load_state_dict(gnss_weights, strict=False)
        print(f"✅ GNSS Encoder 初始权重加载自 Stage1: {gnss_stage1_path}")
    else:
        print(f"⚠️ 未找到 GNSS encoder 权重: {gnss_stage1_path}")

    # --- 冻结视觉 / IMU，不训练 ---
    for name, param in model.named_parameters():
        if "gnss_encoder" in name or "transformer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.L1Loss()

    # --- 训练并保存 ---
    checkpoint_dir = "E:/VITF/gnss_stage2"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, 11):
        model.train()
        epoch_loss = 0.0

        for (imgs, imus, gnss, _, _), gts in tqdm(loader, desc=f"[Stage 2][Epoch {epoch}]"):
            gnss, gts = gnss.cuda().float(), gts.cuda().float()

            optimizer.zero_grad()
            pred = model(gnss)
            loss = criterion(pred, gts)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"[Stage 2][Epoch {epoch}] Avg Loss: {avg_loss:.6f}")

        # 保存模型
        save_path = os.path.join(checkpoint_dir, f"stage2_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"✅ 模型已保存: {save_path}")

# ===== 执行入口 =====
if __name__ == "__main__":
    main()
