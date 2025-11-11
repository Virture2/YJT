import sys
sys.path.insert(0, '../')
from src.data.components.new_KITTI_dataset import KITTI
import torch
from src.utils import new_custom_transform

# 定义transform
transform_train = [new_custom_transform.ToTensor(),
                   new_custom_transform.Resize((256, 512))]
transform_train = new_custom_transform.Compose(transform_train)

# 加载Dataset
dataset = KITTI(
    "kitti_data",
    train_seqs=['05','07','10'],
    transform=transform_train,
    sequence_length=11
)
save_dir = "kitti_latent_data/val_10"
# save_dir = "kitti_latent_data/vift_val"


loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Helper类
class ObjFromDict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

# 加载Encoder
from src.models.components.new_vsvio import Encoder
params = {
    "img_w": 512,
    "img_h": 256,
    "v_f_len": 512,
    "i_f_len": 256,
    "g_f_len": 128,      # GNSS输出特征维度
    "imu_dropout": 0.1,
    "seq_len": 11
}
params = ObjFromDict(params)

# FeatureEncodingModel
class FeatureEncodingModel(torch.nn.Module):
    def __init__(self, params):
        super(FeatureEncodingModel, self).__init__()
        self.Feature_net = Encoder(params)

    def forward(self, imgs, imus, gnss):
        feat_v, feat_i, feat_g = self.Feature_net(imgs, imus, gnss)
        return feat_v, feat_i, feat_g

model = FeatureEncodingModel(params)

# ======== 第一次加载基础预训练权重 (视觉+IMU) =========
pretrained_w = torch.load("../pretrained_models/vf_512_if_256_3e-05.model", map_location='cpu')
model_dict = model.state_dict()
update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
print(f"加载了 {len(update_dict)} 个视觉/IMU权重")
model_dict.update(update_dict)
model.load_state_dict(model_dict)

# ======== 再加载你新训练的GNSS权重 =========
# 注意：这里路径用你实际保存的GNSS权重路径
gnss_trained_weights = torch.load(r"E:\VITF\pretrained_models\gnss_encoder_trained.pth", map_location='cpu')
model_dict = model.state_dict()
# 只保留GNSS分支权重
gnss_update = {k: v for k, v in gnss_trained_weights.items() if "gnss_encoder" in k}
# gnss_update = {
#     "Feature_net." + k: v
#     for k, v in gnss_trained_weights.items()
#     if "gnss_encoder" in k
# }

print(f"加载了 {len(gnss_update)} 个GNSS分支权重")
model_dict.update(gnss_update)
model.load_state_dict(model_dict)

# 如果你想冻结视觉+IMU，这里可以保留，否则删掉
# if "gnss_encoder" not in name:
for name, param in model.Feature_net.named_parameters():
    param.requires_grad = False

#######################################################

# 创建保存目录
import os
os.makedirs(save_dir, exist_ok=True)

# 保存latent
import numpy as np
from tqdm import tqdm

model.eval()
model.to("cuda")

with torch.no_grad():
    for i, ((imgs, imus, gnss, rot, w), gts) in tqdm(enumerate(loader), total=len(loader)):
        imgs = imgs.to("cuda").float()
        imus = imus.to("cuda").float()
        gnss = gnss.to("cuda").float()   # 送入模型

        # 提取三个特征
        feat_v, feat_i, feat_g = model(imgs, imus, gnss)

        # 拼接
        latent_vector = torch.cat((feat_v, feat_i, feat_g), dim=2)
        # latent_vector = torch.cat((feat_v, feat_i), dim=2)
        latent_vector = latent_vector.squeeze(0)

        # 保存
        np.save(os.path.join(save_dir, f"{i}.npy"), latent_vector.cpu().numpy())
        np.save(os.path.join(save_dir, f"{i}_gt.npy"), gts.cpu().numpy())
        np.save(os.path.join(save_dir, f"{i}_rot.npy"), rot.cpu().numpy())
        np.save(os.path.join(save_dir, f"{i}_w.npy"), w.cpu().numpy())
        # np.save(os.path.join(save_dir, f"{i}_gnss.npy"), gnss.cpu().numpy())


# import sys
# sys.path.insert(0, '../')
# import os
# import torch
# import numpy as np
# from tqdm import tqdm
#
# # ===== 数据集和 transform =====
# from src.data.components.new_KITTI_dataset import KITTI
# from src.utils import new_custom_transform
#
# # 定义transform
# transform_train = [new_custom_transform.ToTensor(),
#                    new_custom_transform.Resize((432, 960))]
# transform_train = new_custom_transform.Compose(transform_train)
#
# # 加载Dataset
# dataset = KITTI(
#     "kitti_data",
#     train_seqs=['05','07','10'],
#     transform=transform_train,
#     sequence_length=11
# )
# save_dir = "kitti_latent_data/val_10"
# os.makedirs(save_dir, exist_ok=True)
#
# loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
#
# # ===== Helper 类 =====
# class ObjFromDict:
#     def __init__(self, dictionary):
#         for k, v in dictionary.items():
#             setattr(self, k, v)
#
# # ===== 加载 Encoder =====
# from src.models.components.new_vsvio import Encoder
# import argparse
# import json
#
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--cfg",
#     help="experiment configure file name",
#     default="sintel-L.json",
#     type=str,
# )
# args = parser.parse_args()
#
# # ===== 读取 cfg JSON 并生成 args 对象 =====
# with open(args.cfg, 'r') as f:
#     raft_cfg = json.load(f)
#
# raft_args = ObjFromDict(raft_cfg)
#
# params = {
#     "img_w": 512,
#     "img_h": 256,
#     "v_f_len": 512,   # RAFT 输出压缩到512维
#     "i_f_len": 256,
#     "g_f_len": 128,   # GNSS输出特征维度
#     "imu_dropout": 0.1,
#     "seq_len": 11
# }
# params = ObjFromDict(params)
#
# class FeatureEncodingModel(torch.nn.Module):
#     def __init__(self, params):
#         super(FeatureEncodingModel, self).__init__()
#         self.Feature_net = Encoder(params, raft_args)
#
#     def forward(self, imgs, imus, gnss):
#         feat_v, feat_i, feat_g = self.Feature_net(imgs, imus, gnss)
#         return feat_v, feat_i, feat_g
#
# model = FeatureEncodingModel(params)
#
# # ===== 加载视觉RAFT分支预训练权重 =====
# raft_pretrained_w = torch.load(
#     "../pretrained_models/FA-C-T-Mixed-TSKH432x960.pth",
#     map_location="cpu"
# )
#
# # 给预训练权重的 key 加上前缀
# raft_pretrained_w = {f"Feature_net.raft.{k}": v for k, v in raft_pretrained_w.items()}
#
# model_dict = model.state_dict()
#
# # 过滤，只保留匹配的 key
# filtered_dict = {k: v for k, v in raft_pretrained_w.items() if k in model_dict}
#
# # 更新并加载
# model_dict.update(filtered_dict)
# model.load_state_dict(model_dict)
#
# print(f"已加载 {len(filtered_dict)} 个匹配的参数（总计 {len(model_dict)} 个）")
# print("已匹配参数示例：", list(filtered_dict.keys())[:10])
#
#
# # ===== 加载预训练权重 =====
# # 1. 只加载IMU分支
# pretrained_w = torch.load("../pretrained_models/vf_512_if_256_3e-05.model", map_location='cpu')
# model_dict = model.state_dict()
# imu_update = {k: v for k, v in pretrained_w.items() if "inertial_encoder" in k}
# print(f"加载了 {len(imu_update)} 个IMU权重")
# model_dict.update(imu_update)
# model.load_state_dict(model_dict)
#
# # 2. 加载GNSS分支权重
# gnss_trained_weights = torch.load(r"E:\VITF\pretrained_models\gnss_encoder_trained.pth", map_location='cpu')
# gnss_update = {k: v for k, v in gnss_trained_weights.items() if "gnss_encoder" in k}
# print(f"加载了 {len(gnss_update)} 个GNSS分支权重")
# model_dict.update(gnss_update)
# model.load_state_dict(model_dict)
#
# # ===== 冻结IMU和GNSS（只训练视觉RAFT） =====
# for name, param in model.Feature_net.named_parameters():
#     param.requires_grad = False
#
# # ===== 提取latent并保存 =====
# model.eval()
# model.to("cuda")
#
# with torch.no_grad():
#     for i, ((imgs, imus, gnss, rot, w), gts) in tqdm(enumerate(loader), total=len(loader)):
#         imgs = imgs.to("cuda").float()
#         imus = imus.to("cuda").float()
#         gnss = gnss.to("cuda").float()
#
#         # 提取三个特征
#         feat_v, feat_i, feat_g = model(imgs, imus, gnss)  # (B, seq_len, dim)
#
#         # 拼接 latent
#         latent_vector = torch.cat((feat_v, feat_i, feat_g), dim=2)  # (B, seq_len, v+i+g)
#         latent_vector = latent_vector.squeeze(0)  # 去掉batch维度
#
#         # 保存
#         np.save(os.path.join(save_dir, f"{i}.npy"), latent_vector.cpu().numpy())
#         np.save(os.path.join(save_dir, f"{i}_gt.npy"), gts.cpu().numpy())
#         np.save(os.path.join(save_dir, f"{i}_rot.npy"), rot.cpu().numpy())
#         np.save(os.path.join(save_dir, f"{i}_w.npy"), w.cpu().numpy())

# import sys
# sys.path.insert(0, '../')
# from src.data.components.KITTI_dataset import KITTI
# import torch
# from src.utils import custom_transform
#
# # 定义transform
# transform_train = [custom_transform.ToTensor(),
#                    custom_transform.Resize((256, 512))]
# transform_train = custom_transform.Compose(transform_train)
#
# # 加载Dataset
# dataset = KITTI(
#     "kitti_data",
#     train_seqs=['05','07','10'],
#     transform=transform_train,
#     sequence_length=11
# )
# save_dir = "kitti_latent_data/val_10"
#
# loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
#
# # Helper类
# class ObjFromDict:
#     def __init__(self, dictionary):
#         for k, v in dictionary.items():
#             setattr(self, k, v)
#
# # 加载Encoder
# from src.models.components.vsvio import Encoder
# params = {
#     "img_w": 512,
#     "img_h": 256,
#     "v_f_len": 512,
#     "i_f_len": 256,
#     "imu_dropout": 0.1,
#     "seq_len": 11
# }
# params = ObjFromDict(params)
#
# # FeatureEncodingModel
# class FeatureEncodingModel(torch.nn.Module):
#     def __init__(self, params):
#         super(FeatureEncodingModel, self).__init__()
#         self.Feature_net = Encoder(params)
#
#     def forward(self, imgs, imus):
#         feat_v, feat_i = self.Feature_net(imgs, imus)
#         return feat_v, feat_i
#
# model = FeatureEncodingModel(params)
#
# # ======== 第一次加载基础预训练权重 (视觉+IMU) =========
# pretrained_w = torch.load("../pretrained_models/vf_512_if_256_3e-05.model", map_location='cpu')
# model_dict = model.state_dict()
# update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
# print(f"加载了 {len(update_dict)} 个视觉/IMU权重")
# model_dict.update(update_dict)
# model.load_state_dict(model_dict)
#
# # 如果你想冻结视觉+IMU，这里可以保留，否则删掉
# for param in model.Feature_net.parameters():
#     param.requires_grad = False
#
#
# # 创建保存目录
# import os
# os.makedirs(save_dir, exist_ok=True)
#
# # 保存latent
# import numpy as np
# from tqdm import tqdm
#
# model.eval()
# model.to("cuda")
#
# with torch.no_grad():
#     for i, ((imgs, imus, rot, w), gts) in tqdm(enumerate(loader), total=len(loader)):
#         imgs = imgs.to("cuda").float()
#         imus = imus.to("cuda").float()
#
#         # 提取三个特征
#         feat_v, feat_i = model(imgs, imus)
#
#         # 拼接
#         latent_vector = torch.cat((feat_v, feat_i), dim=2)
#         latent_vector = latent_vector.squeeze(0)
#
#         # 保存
#         np.save(os.path.join(save_dir, f"{i}.npy"), latent_vector.cpu().numpy())
#         np.save(os.path.join(save_dir, f"{i}_gt.npy"), gts.cpu().numpy())
#         np.save(os.path.join(save_dir, f"{i}_rot.npy"), rot.cpu().numpy())
#         np.save(os.path.join(save_dir, f"{i}_w.npy"), w.cpu().numpy())
