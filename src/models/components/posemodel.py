from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
# from ...utils.feature_similarity import compute_feature_similarity

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0.0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )

# The inertial encoder for raw imu data
class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()
        # self.training = False
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
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
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256* 1 * 11, opt.i_f_len)
        # self.eval()

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        # x=x.permute(0, 2, 1)
        # for i in range(len(self.encoder_conv)):
        #     x = self.encoder_conv[i](x)
        #     print(i,x)
        x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))                   # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, 256)
    # def train(self, mode):
    #     return self

# The inertial encoder for imupreintegration data
class Preintegration_Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Preintegration_Inertial_encoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=3, padding=1),
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
            nn.Dropout(opt.imu_dropout),#)
            # new
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256, opt.i_f_len)

    def forward(self, x):
        # x: (N, seq_len, , 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N, 64, seq_len)
        out = self.proj(x.permute(0, 2, 1))                   # out: (N, seq_len, i_f_len)
        return out

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        # CNN
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)
        self.inertial_encoder = Inertial_encoder(opt)

    def forward(self, img, imu):
        # v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        v = img
        batch_size = v.size(0)
        seq_len = v.size(1)

        # image CNN
        # v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        # v = self.encode_image(v)
        # v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        # v = self.visual_head(v)  # (batch, seq_len, 256)
        
        # IMU CNN
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)
        return v, imu

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


class PoseTransformer(nn.Module):
    def __init__(self, params, input_dim=768+64, embedding_dim=768+64, num_layers=4, nhead=8, dim_feedforward=128, dropout=0.0):
        super(PoseTransformer, self).__init__()

        self.params = params
        self.v_i_embed = params.i_f_len + params.v_f_len
        self.v_i_g_embed = params.i_f_len + params.v_f_len + params.g_f_len
        input_dim = self.v_i_g_embed
        nhead = params.num_heads
        dim_feedforward = params.hidden_dim
        dropout = params.dropout
        
        self.embedding_dim = params.embed_dim
        self.num_layers = params.num_layers

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
            num_layers=params.num_layers
        )
        # Add the fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6))
        # self.fc2_translation = nn.Sequential(
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(self.embedding_dim, 3))
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def forward(self, visual_inertial_features, gps_features):
        if self.params.g_f_len != 0:
            visual_inertial_gps_features = torch.cat((visual_inertial_features, gps_features), dim=2)
        else:
            visual_inertial_gps_features = visual_inertial_features
        
        #visual_inertial_features, _, _ = batch
        seq_length = visual_inertial_gps_features.size(1)
        
        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_gps_features.device)
        visual_inertial_gps_features = self.fc1(visual_inertial_gps_features)
        visual_inertial_gps_features += pos_embedding

        
        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_gps_features.device)
        output = self.transformer_encoder(visual_inertial_gps_features, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc2(output)
        # output_translation = self.fc2_translation(output)
        # output = torch.cat((output_rotation, output_translation), dim=-1)
        return output
        
class FeatureEncodingModel(torch.nn.Module):
    def __init__(self, params):
        super(FeatureEncodingModel, self).__init__()
        self.Feature_net = Encoder(params)
    def forward(self, imgs, imus):
        feat_v, feat_i = self.Feature_net(imgs, imus)
        return feat_v, feat_i

class PoseModel(nn.Module):
    def __init__(self,params):
        super(PoseModel, self).__init__()
        self.params = params
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.params.use_preload_vi:
            # if not params.preload_img_feature:
            #     self.Feature_net = Encoder(params) # 为了匹配相应的键值，加载预训练权重
            # else:
            #     self.Feature_net = nn.Identity()
            self.Feature_net = Encoder(params)
            self.fusion_net = PoseTransformer(params)

        if params.use_gps_net:
            # self.gps_feature =
            # params = params._replace(g_f_len = )
            if self.params.use_gps_z_and_covariance:
                g_f_len_init = 6
            else:
                g_f_len_init = 3
            # self.gps_feature = nn.Sequential(nn.Linear(g_f_len_init, int(params.g_f_len / 2)),
            #                     nn.LeakyReLU(0.1, inplace=True), nn.BatchNorm1d(int(params.g_f_len / 2)),
            #                                  nn.Linear(int(params.g_f_len / 2), params.g_f_len),)
            self.gps_feature = nn.Sequential(nn.Linear(g_f_len_init, int(params.g_f_len / 2)),
                                nn.LeakyReLU(0.1, inplace=True), nn.Linear(int(params.g_f_len / 2), 256),
                                nn.Linear(256, 256), nn.LeakyReLU(0.1, inplace=True), nn.Linear(256, params.g_f_len))
        else:
            self.gps_feature = nn.Identity()

    def forward(self, vi, gps):#, hc=None):#, attention):
        # 提特征
        if not self.params.use_preload_vi:
            imgs, imupreintegrations = vi
            feature_v, feature_i = self.Feature_net(imgs,imupreintegrations)
            vi = torch.cat((feature_v,feature_i),dim=2)
        #     feature_imu = self.imu_feature(imupreintegrations)
        #     #feature_imu,hc = self.imupose(imupreintegrations,hc)
        #     vi = torch.cat((feature_v,feature_imu), dim=2)
        #     #imuoutput,hc = self.imupose(imupreintegrations,hc)

        gps_xyz = gps[:, :-1, :3] - gps[:, 1:, :3] # (batch, context_len-1, 3)*2->(batch, context_len-1, 6)
        if self.params.use_gps_z_and_covariance:
            gps = torch.cat((gps_xyz,gps[:, 1:, 3:]), dim=2)
        else:
            gps = gps_xyz[:, :, :3]
        feature_gps = self.gps_feature(gps)
        

        #fused_embed_rotation, fused_embed_translation = self.fusion_net(vi, feature_gps)#, attention)
        out_ref = self.fusion_net(vi, feature_gps)

        return out_ref

class PoseTransformer_gps(nn.Module):
    def __init__(self, params, input_dim=768+64, embedding_dim=768+64, num_layers=4, nhead=8, dim_feedforward=128, dropout=0.0):
        super(PoseTransformer_gps, self).__init__()

        self.params = params
        self.v_i_embed = params.i_f_len + params.v_f_len
        self.v_i_g_embed = params.i_f_len + params.v_f_len + params.g_f_len
        input_dim = self.v_i_g_embed
        nhead = params.num_heads
        dim_feedforward = params.hidden_dim
        dropout = params.dropout
        
        self.embedding_dim = params.embed_dim
        self.num_layers = params.num_layers

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
        # Add the fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6))
        # self.fc2_translation = nn.Sequential(
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(self.embedding_dim, 3))
    
        if self.params.g_f_len != 0:
            if self.params.use_gps_z_and_covariance:
                g_f_len_init = 6
            else:
                g_f_len_init = 3
            self.gps_feature = nn.Sequential(nn.Linear(g_f_len_init, int(params.g_f_len / 2)),
                                nn.LeakyReLU(0.1, inplace=True), nn.Linear(int(params.g_f_len / 2), 256),
                                nn.Linear(256, 256), nn.LeakyReLU(0.1, inplace=True), nn.Linear(256, params.g_f_len))
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def forward(self, visual_inertial_features, gps):
        if self.params.g_f_len != 0:
            gps_xyz = gps[:, :-1, :3] - gps[:, 1:, :3] # (batch, context_len-1, 3)*2->(batch, context_len-1, 6)
            if self.params.use_gps_z_and_covariance:
                gps = torch.cat((gps_xyz,gps[:, 1:, 3:]), dim=2)
            else:
                gps = gps_xyz[:, :, :3]
            gps_features = self.gps_feature(gps)
            visual_inertial_gps_features = torch.cat((visual_inertial_features, gps_features), dim=2)
        else:
            visual_inertial_gps_features = visual_inertial_features
        
        #visual_inertial_features, _, _ = batch
        seq_length = visual_inertial_gps_features.size(1)
        
        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_gps_features.device)
        visual_inertial_gps_features = self.fc1(visual_inertial_gps_features)
        visual_inertial_gps_features += pos_embedding

        
        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_gps_features.device)
        output = self.transformer_encoder(visual_inertial_gps_features, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc2(output)
        # output_translation = self.fc2_translation(output)
        # output = torch.cat((output_rotation, output_translation), dim=-1)
        return output

class TemporalEncodingModule(nn.Module):
    """
    时间编码模块，用于生成考虑IMU和相机特征偏移的相对位置编码
    """
    def __init__(self, seq_len=11, max_offset=0.5, embed_dim=512):
        """
        初始化时间编码模块
        
        参数:
            seq_len: 序列长度
            max_offset: 可学习偏移的最大范围，默认±0.5
        """
        super().__init__()
        
        self.seq_len = seq_len
        # self.num_heads = num_heads
        
        # 可学习的相对位置偏差表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(seq_len, embed_dim)
        )
        
        # 可学习的偏移参数，通过tanh限制在[-max_offset, max_offset]
        # self.learnable_offset = nn.Parameter(torch.zeros(1))
        self.max_offset = float(max_offset)
        self.offset_encoder = nn.Sequential(
            nn.Linear(self.seq_len * self.seq_len, 64),  
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
        )
        
        # 预计算相对位置索引
        self.register_buffer(
            "relative_position_index",
            self._create_relative_position_index()
        )
        
    def _create_relative_position_index(self):
        """创建相对位置索引矩阵"""
        coords = torch.arange(self.seq_len)
        # relative_coords = coords[:, None] - coords[None, :]
        # relative_coords += self.seq_len - 1  # 转换为非负索引
        # return relative_coords
        return coords

    def _temporal_interpolation(self, features, source_timestamps, target_timestamps):
        """
        时间插值函数，将source_timestamps时刻的features插值到target_timestamps时刻
        
        参数:
            features: [batch_size, seq_len, embed_dim]
            source_timestamps: [seq_len]
            target_timestamps: [seq_len]
        """
        batch_size, seq_len, embed_dim = features.shape
        
        # 1. 将时间戳归一化到[-1,1]区间（适合grid_sample）
        min_time = source_timestamps.min()
        max_time = source_timestamps.max()
        norm_source_timestamps = 2 * (source_timestamps - min_time) / (max_time - min_time) - 1
        norm_target_timestamps = 2 * (target_timestamps - min_time) / (max_time - min_time) - 1
        
        # 2. 构建采样网格 [batch_size, 1, seq_len, 1]
        grid = norm_target_timestamps.unsqueeze(0).unsqueeze(1).unsqueeze(-1) # 单独变量
        # grid = norm_target_timestamps.unsqueeze(1).unsqueeze(-1)              # attention-based 
        grid = grid.expand(batch_size, 1, seq_len, 1)  # 扩展批次维度
        
        # 3. 添加y坐标（全0）以满足grid_sample要求 [batch_size, 1, seq_len, 2]
        grid = torch.cat([grid, torch.zeros_like(grid)], dim=-1)
        
        # 4. 重塑features为[batch_size, embed_dim, 1, seq_len]
        features_reshaped = features.permute(0, 2, 1).unsqueeze(2)
        
        # 5. 使用双线性插值进行时间重采样
        interpolated_features = F.grid_sample(
            features_reshaped, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        )  # [batch_size, embed_dim, 1, seq_len]
        
        # 6. 恢复原始形状 [batch_size, seq_len, embed_dim]
        return interpolated_features.squeeze(2).permute(0, 2, 1)
    
    def forward(self, feature, attention):
        """
        生成时间编码矩阵
            
        返回:
            时间编码矩阵 [1, num_heads, seq_len, seq_len]
        """
        T = self.seq_len

        # 应用门控函数限制偏移范围
        # bounded_offset = torch.tanh(self.learnable_offset) * self.max_offset # [1]
        bounded_offset = torch.tanh(self.offset_encoder(attention.reshape(attention.size(0), -1))) * self.max_offset # [batch_size, 1]
        # print(bounded_offset)
        # 调整相对位置索引并添加偏移
        adjusted_index = self.relative_position_index.float() + (-1) #bounded_offset

        # 使用双线性插值进行时间重采样
        aligned_visual_features = self._temporal_interpolation(
            feature, 
            self.relative_position_index,
            adjusted_index
        )
        return aligned_visual_features
        
        # # 确保索引在有效范围内 [0, T - 1]
        # adjusted_index = torch.clamp(adjusted_index, 0, T - 1)
        
        # # 计算相邻的整数索引（下取整和上取整）
        # lower_idx = torch.floor(adjusted_index).long()
        # upper_idx = torch.ceil(adjusted_index).long()
        
        # # 计算插值权重（小数部分，表示距离lower_idx的比例）
        # weight = adjusted_index - lower_idx.float()
        
        # # 从相对位置偏差表中获取相邻位置的值
        # # [seq_len, seq_len, num_heads]
        # lower_val = self.relative_position_bias_table[lower_idx]
        # # [seq_len, seq_len, num_heads]
        # upper_val = self.relative_position_bias_table[upper_idx]
        
        # # 线性插值公式：lower_val * (1-weight) + upper_val * weight
        # bias = lower_val * (1 - weight.unsqueeze(-1)) + upper_val * weight.unsqueeze(-1)

        # origina_temporal_pos = self.relative_position_bias_table[self.relative_position_index]

        # return bias, origina_temporal_pos
         # 调整维度为 [1, seq_len]
        # return bias.unsqueeze(0), origina_temporal_pos.unsqueeze(0)

class TemporalEncodingModule_inattention(nn.Module):
    """
    时间编码模块，用于生成考虑IMU和相机特征偏移的相对位置编码
    """
    def __init__(self, seq_len=11, num_heads=8, max_offset=0.5):
        """
        初始化时间编码模块
        
        参数:
            seq_len: 序列长度
            num_heads: 注意力头数
            max_offset: 可学习偏移的最大范围，默认±0.5
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.num_heads = num_heads
        
        # 可学习的相对位置偏差表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * seq_len - 1, num_heads)
        )
        
        # 可学习的偏移参数，通过tanh限制在[-max_offset, max_offset]
        self.learnable_offset = nn.Parameter(torch.zeros(1))
        self.max_offset = float(max_offset)
        
        # 预计算相对位置索引
        self.register_buffer(
            "relative_position_index",
            self._create_relative_position_index()
        )
        
    def _create_relative_position_index(self):
        """创建相对位置索引矩阵"""
        coords = torch.arange(self.seq_len)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += self.seq_len - 1  # 转换为非负索引
        return relative_coords
    
    def forward(self): #C
        """
        生成时间编码矩阵
        
        参数:
            C: 特征通道数
            
        返回:
            时间编码矩阵 [1, num_heads, seq_len, seq_len]
        """
        T = self.seq_len

        # 应用门控函数限制偏移范围
        bounded_offset = torch.tanh(self.learnable_offset) * self.max_offset
        
        # 调整相对位置索引并添加偏移
        adjusted_index = self.relative_position_index.float() + bounded_offset
        
        # 确保索引在有效范围内 [0, 2*T-2]
        adjusted_index = torch.clamp(adjusted_index, 0, 2 * T - 2)
        
        # 计算相邻的整数索引（下取整和上取整）
        lower_idx = torch.floor(adjusted_index).long()
        upper_idx = torch.ceil(adjusted_index).long()
        
        # 计算插值权重（小数部分，表示距离lower_idx的比例）
        weight = adjusted_index - lower_idx.float()
        
        # 从相对位置偏差表中获取相邻位置的值
        # [seq_len, seq_len, num_heads]
        lower_val = self.relative_position_bias_table[lower_idx]
        # [seq_len, seq_len, num_heads]
        upper_val = self.relative_position_bias_table[upper_idx]
        
        # 线性插值公式：lower_val * (1-weight) + upper_val * weight
        bias = lower_val * (1 - weight.unsqueeze(-1)) + upper_val * weight.unsqueeze(-1)

         # 调整维度为 [1, num_heads, seq_len, seq_len]
        return bias.permute(2, 0, 1).unsqueeze(0)

        # 使用双线性插值获取更准确的相对位置偏差   #不能直接使用grid_sample，有错
        # relative_position_bias = F.grid_sample(
        #     self.relative_position_bias_table.unsqueeze(0).unsqueeze(0),  # [1, 1, 2*T-1, num_heads]
        #     adjusted_index.view(1, T, T, 1).unsqueeze(0),  # [1, T, T, 1, 1]
        #     mode='bilinear', padding_mode='border'
        # ).squeeze(0).squeeze(0).permute(2, 0, 1)  # [num_heads, T, T]
        
        # # 调整维度并扩展
        # relative_position_bias = relative_position_bias[None].permute(3, 0, 1, 2).expand(
        #     self.num_heads, (C // self.num_heads) * (C // self.num_heads), T, T
        # )
        
        # # 插值和像素洗牌操作
        # relative_position_bias = F.interpolate(relative_position_bias, scale_factor=1/11)
        # relative_position_bias = F.pixel_shuffle(
        #     relative_position_bias, upscale_factor=int(C // self.num_heads)
        # ).permute(1, 0, 2, 3)
        
         # 调整维度为 [1, num_heads, seq_len, seq_len]
        # return relative_position_bias.unsqueeze(0)
    
class CustomTemporalAttention(nn.Module):
    """
    自定义时间感知注意力层，在标准注意力机制中集成时间编码
    """
    def __init__(self, embed_dim=256, num_heads=8, seq_len=11, max_offset=0.5, 
                 dropout=0.0, qkv_bias=True, use_temporal_encoding_inattention = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            print('Embed_dem cannot divide num_ heads evenly')
            raise
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_temporal_encoding_inattention = use_temporal_encoding_inattention
        
        # 线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 时间编码模块
        if self.use_temporal_encoding_inattention:
            self.temporal_encoding = TemporalEncodingModule_inattention(
                seq_len=seq_len,
                num_heads=num_heads,
                max_offset=max_offset
            )
        
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, is_causal=True):
        """
        前向传播
        
        参数:
            query: [batch_size, seq_len, embed_dim]
            key: [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            attn_mask: 可选的注意力掩码 [batch_size, seq_len, seq_len]
            key_padding_mask: 可选的键填充掩码 [batch_size, seq_len]
        """
        batch_size, seq_len, _ = query.shape
        
        # 线性投影并分割多头
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力分数 [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if self.use_temporal_encoding_inattention:
            # 获取时间编码矩阵 [1, num_heads, seq_len, seq_len]
            temporal_bias = self.temporal_encoding()
        
            # 添加时间编码到注意力分数
            attn_scores = attn_scores + temporal_bias
        
        # 应用注意力掩码（如果有）,默认为因果，未处理非因果。未考虑掩码对时间的影响！！！！！！！
        if attn_mask is not None:
            # 扩展掩码以匹配多头维度
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attn_scores = attn_scores + attn_mask # 扩展到多头维度
            
        # 应用键填充掩码（如果有）
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9
            )
        
        # 应用softmax和dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        # 计算注意力输出
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 最终投影
        output = self.resid_drop(self.out_proj(output))
        
        return output
    
class TemporalTransformerEncoderLayer(nn.Module):
    """
    使用自定义时间感知注意力的Transformer层
    """
    def __init__(self, embed_dim=256, num_heads=8, seq_len=11, max_offset=0.5,
                 hidden_dim=128, dropout=0.0, qkv_bias=True, layer_norm_eps=1e-5, use_temporal_encoding_inattention = True):
        super().__init__()
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # 自定义时间感知注意力
        self.attn = CustomTemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            max_offset=max_offset,
            dropout=dropout,
            qkv_bias=qkv_bias,
            use_temporal_encoding_inattention = use_temporal_encoding_inattention,
        )
        
        # 层归一化和MLP
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        mlp_hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, is_causal=True):
        """
        前向传播
        
        参数:
            x: [batch_size, seq_len, embed_dim]
            attn_mask: 可选的注意力掩码
            key_padding_mask: 可选的键填充掩码
        """
        # 自注意力和残差连接
        x = v + self.attn(
            query=self.norm1(q),
            key=self.norm1(k),
            value=self.norm1(v),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal
        )
        
        # MLP和残差连接
        x = x + self.mlp(self.norm2(x))
        
        return x

class PoseTemporalTransformerEncoder(nn.Module):
    """
    集成时间编码的Transformer编码器，使用PyTorch的nn.TransformerEncoder
    """
    def __init__(self, params, embed_dim=256, num_heads=8, num_layers=4, seq_len=10, max_offset=0.5,
                 hidden_dim=128, dropout=0.0, layer_norm_eps=1e-5, qkv_bias=True, use_temporal_encoding_inattention=False, use_temporal_encoding=True, batch_first=True): # activation="gelu"
        super().__init__()

        self.params = params
        # 创建编码器层
        input_dim = self.params.i_f_len + self.params.v_f_len + self.params.g_f_len
        self.embed_dim = self.params.embed_dim

        # self.fc1 = nn.Sequential(
        #     nn.Linear(input_dim, self.params.embed_dim),
        # )
        self.fc1_v = nn.Sequential(
            nn.Linear(self.params.v_f_len, self.params.embed_dim),
        )
        self.fc1_i = nn.Sequential(
            nn.Linear(self.params.i_f_len, self.params.embed_dim),
        )
        
        # 创建多层编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=TemporalTransformerEncoderLayer(
                embed_dim=self.params.embed_dim,
                num_heads=self.params.num_heads,
                seq_len=self.params.seq_len,
                max_offset=self.params.max_offset,
                hidden_dim=self.params.hidden_dim,
                dropout=self.params.dropout,
                #activation=activation,
                layer_norm_eps=self.params.layer_norm_eps,
                qkv_bias=self.params.qkv_bias, 
                use_temporal_encoding_inattention=self.params.use_temporal_encoding_inattention,
            ),
            num_layers=self.params.num_layers,
            norm=nn.LayerNorm(self.params.embed_dim, eps=self.params.layer_norm_eps)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.params.embed_dim, self.params.embed_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.params.embed_dim, 6))
        
        self.batch_first = self.params.batch_first

        if self.params.use_temporal_encoding:
            self.temporal_encoding = TemporalEncodingModule(
                seq_len=self.params.seq_len,
                max_offset=self.params.max_offset,
                embed_dim=self.embed_dim,
            )
            self.base_cross_attn = nn.MultiheadAttention(self.params.embed_dim, self.params.num_heads, batch_first=self.params.batch_first)

    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(math.log(10000.0) / self.embed_dim))
        pos_embedding = torch.zeros(seq_length, self.embed_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )
        
    def forward(self, batch, target, mask=None, src_key_padding_mask=None): #记得删target
        """
        前向传播
        
        参数:
            src: 输入序列，形状为 (batch, seq_len, embed_dim) 或 (seq_len, batch, embed_dim)
            mask: 可选的注意力掩码
            src_key_padding_mask: 可选的键填充掩码
        """
        # 确保输入格式正确
        visual_inertial_gps_features, _, _ = batch
        if not self.batch_first:
            visual_inertial_gps_features = visual_inertial_gps_features.transpose(0, 1)  # 转换为 (batch, seq_len, embed_dim)
            seq_length = visual_inertial_gps_features.size(0)
        else:
            seq_length = visual_inertial_gps_features.size(1)

        visual_feature = visual_inertial_gps_features[:, :, :self.params.v_f_len]
        inertial_feature = visual_inertial_gps_features[:, :, self.params.v_f_len:self.params.v_f_len+self.params.i_f_len]

        # print("visual_inertial_gps_features shape:", visual_inertial_gps_features.shape)
        # print("visual_feature shape:", visual_feature.shape)
        # print("inertial_feature shape:", inertial_feature.shape)

        visual_feature = self.fc1_v(visual_feature)
        inertial_feature = self.fc1_i(inertial_feature)
        # a=compute_feature_similarity(inertial_feature.squeeze(0), visual_feature.squeeze(0), method='cosine', device=inertial_feature.device)
        # print(a, a.shape)

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_gps_features.device)
        # visual_inertial_gps_features = self.fc1(visual_inertial_gps_features)
        # visual_inertial_gps_features += pos_embedding
        visual_feature += pos_embedding
        inertial_feature += pos_embedding
        
        # Passing through the transformer encoder with the mask # 通过Transformer编码器
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_gps_features.device)
        # output = self.transformer_encoder(visual_inertial_gps_features, mask=mask, is_causal=True)

        if self.params.use_temporal_encoding:
            # 计算基础注意力权重（用于引导对齐）
            _, base_attn_weights = self.base_cross_attn(
                query=inertial_feature,
                key=visual_feature,
                value=visual_feature
            )  # [batch_size, seq_len, seq_len]
            # 获取时间编码矩阵 [1, seq_len, embeding_dim]
            # temporal_bias, temporal_original = self.temporal_encoding(base_attn_weights)
            # inertial_feature = inertial_feature + temporal_original
            # visual_feature = visual_feature + temporal_bias

            # temporal_inertial_feature = self.temporal_encoding(inertial_feature)
            temporal_inertial_feature = self.temporal_encoding(inertial_feature, base_attn_weights)
        # 为每个编码器层应用交叉注意力
        output = inertial_feature
        for layer in self.transformer_encoder.layers:
            output = layer(
                output,
                visual_feature,
                visual_feature,
                attn_mask=mask,
                is_causal=True,
            )
        #output = self.transformer_encoder.norm(output)

        output = self.fc2(output)
        # 恢复原始格式（如果需要）
        if not self.batch_first:
            output = output.transpose(0, 1)
        # print("output shape:", output.shape)
        return output

# class IMUSimilaritySampler(nn.Module):
#     def __init__(self, imu_encoder, imu_dim=256, visual_dim=512, embed_dim=512,
#                  window_size=11, input_length=111, output_length=101, temperature=0.1):
#         """
#         固定窗口相似度匹配采样器
#
#         参数:
#             imu_dim: IMU数据维度
#             visual_dim: 视觉特征维度
#             embed_dim: 嵌入维度
#             window_size: 用于相似度计算的窗口大小(10)
#             input_length: 输入IMU序列长度
#             output_length: 输出IMU序列长度
#         """
#         super().__init__()
#         self.imu_dim = imu_dim
#         self.visual_dim = visual_dim
#         self.embed_dim = embed_dim
#         self.window_size = window_size
#         self.input_length = input_length
#         self.output_length = output_length
#         self.max_start_idx = input_length - output_length  # 10
#         self.temperature = temperature
#
#         self.imu_encoder = imu_encoder
#         # IMU特征嵌入网络
#         self.imu_embedder = nn.Sequential(
#             nn.Linear(imu_dim, embed_dim),
#             #nn.LayerNorm(embed_dim),
#             nn.LeakyReLU(0.1),
#             nn.Linear(embed_dim, embed_dim),
#         )
#
#         # 视觉特征嵌入网络
#         self.visual_embedder = nn.Sequential(
#             nn.Linear(visual_dim, embed_dim),
#             #nn.LayerNorm(embed_dim),
#             nn.LeakyReLU(0.1),
#             nn.Linear(embed_dim, embed_dim),
#         )
#
#     def compute_similarity(self, imu_window, visual_feature):
#         """
#         计算IMU窗口与视觉特征的相似度
#
#         参数:
#             imu_window: [batch_size, seq_len*11, imu_dim]
#             visual_feature: [batch_size, seq_len, visual_dim]
#
#         返回:
#             similarity: 相似度得分 [batch_size]
#         """
#         # 嵌入IMU窗口和视觉特征
#         seq_len = visual_feature.shape[1] # 1
#         imu_window = torch.cat([imu_window[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
#         imu_window = self.imu_encoder(imu_window)
#         imu_embedded = self.imu_embedder(imu_window)  # [batch_size, seq_len, embed_dim]
#         visual_embedded = self.visual_embedder(visual_feature)  # [batch_size, embed_dim]
#
#         # 计算视觉特征与IMU窗口各点的相似度
#         visual_embedded_expanded = visual_embedded  # [batch_size, 1, embed_dim]
#         # similarity = torch.bmm(
#         #     visual_embedded_expanded,
#         #     imu_embedded.permute(0, 2, 1)
#         # ).squeeze(-1).squeeze(-1)  # [batch_size]
#         similarity = F.cosine_similarity(imu_embedded.squeeze(1), visual_embedded_expanded.squeeze(1), dim=1) # 使用cosine_similarity计算余弦相似度
#         # 平均相似度得分
#         # similarity = sim_matrix.mean(dim=1)  # [batch_size]
#
#         return similarity
#
#     def forward(self, raw_imu_data, visual_feature):
#         """
#         参数:
#             raw_imu_data: [batch_size, input_length, imu_dim]
#             visual_features: [batch_size, seq_len, visual_dim]
#         """
#         batch_size = raw_imu_data.shape[0]
#
#         # 获取第一个视觉特征 (假设它对应时间序列的开始)
#         #visual_feature = visual_feature[:, :1]  # [batch_size, 1, visual_dim]
#
#         # 1. 计算所有可能窗口的相似度
#         similarities = []
#         for i in range(self.max_start_idx + 1):  # 0到10
#             # 提取IMU窗口
#             imu_window = raw_imu_data[:, i:i+self.window_size]  # [batch_size, window_size, 6]
#
#             # 计算相似度
#             sim = self.compute_similarity(imu_window, visual_feature)
#             similarities.append(sim.unsqueeze(1))  # [batch_size, 1]
#         similarities = torch.cat(similarities, dim=1)  # [batch_size, 11]
#
#         # similarities = F.log_softmax(similarities, dim=1)  # 归一化为log概率分布，作为Gumbel输入
#         # # 2. 使用Gumbel-softmax得到概率分布（可导）
#         # gumbel_probs = F.gumbel_softmax(
#         #     logits=similarities,
#         #     tau=self.temperature if self.training else 1e-3,  # 温度参数，τ越小越接近one-hot，训练时可设为0.1~1.0 测试时温度接近0
#         #     hard=True, #not self.training,  # 训练时用soft选择（hard=False），测试时用hard选择（hard=True)
#         #     eps=1e-10
#         # )  # [batch_size, 11]，每行为近似one-hot的概率分布
#
#         # Straight through.
#         index = similarities.max(dim=-1, keepdim=True)[1]
#         y_hard = torch.zeros_like(
#             similarities, memory_format=torch.legacy_contiguous_format
#         ).scatter_(-1, index, 1.0)
#         gumbel_probs = y_hard - similarities.detach() + similarities
#
#         # 3. 用概率分布加权组合所有候选IMU窗口（关键：保证可导）
#         # 先整理所有候选窗口为[batch_size, num_candidates, window_size, 6]
#         candidate_windows = []
#         for i in range(self.max_start_idx + 1):
#             window = raw_imu_data[:, i:i+self.output_length]  # [batch_size, window_size, 6]
#             candidate_windows.append(window.unsqueeze(1))  # [batch_size, 1, window_size, 6]
#         candidate_windows = torch.cat(candidate_windows, dim=1)  # [batch_size, 11, window_size, 6]
#
#         # 加权组合：概率分布gumbel_probs与候选窗口相乘后求和
#         # gumbel_probs形状为[batch_size, 11, 1, 1]（扩展维度以匹配窗口维度）
#         # print(similarities)
#         # print(gumbel_probs)
#         a, b = torch.max(gumbel_probs, dim=1)
#         print(b)
#         selected_imu_data = (gumbel_probs.unsqueeze(-1).unsqueeze(-1) * candidate_windows).sum(dim=1)
#         # print(selected_imu_data)
#         # 输出：[batch_size, window_size, 6]，与硬选择结果一致（τ→0时）
#
#         # 计算对比损失（仅训练时）
#         #contrastive_loss = torch.tensor(0.0, device=raw_imu_data.device)
#         if self.training: #and true_start_idx is not None:
#             # 正样本相似度：每个样本的真实窗口相似度 [batch_size]
#             positive_sim = similarities[:, true_start_idx]
#
#             # 负样本相似度：每个样本的非真实窗口最大相似度 [batch_size]
#             # 先将真实窗口的相似度设为-∞，再取最大值
#             mask = torch.eye(self.max_start_idx + 1, device=raw_imu_data.device)[true_start_idx]  # [batch_size, 11]
#             negative_sim = (similarities * (1 - mask) - mask * 1e9).max(dim=1)[0]
#
#             # 三元组损失：确保正样本相似度 > 负样本相似度 + margin
#             contrastive_loss = F.relu(negative_sim + self.margin - positive_sim).mean()
#
#         # # 2. 选择相似度最大的起始位置 梯度反传是否有问题？
#         # max_similarities, start_indices = torch.max(similarities, dim=1)  # [batch_size]
#         # # start_indices = torch.zeros_like(start_indices) # 对比
#         # # start_indices = torch.full_like(start_indices, 10)
#         # print(start_indices)
#
#         # # 3. 根据起始位置提取连续的110个IMU数据点
#         # selected_imu_data = []
#         # for b in range(batch_size):
#         #     start_idx = start_indices[b].item()
#         #     # print(start_idx)
#         #     selected = raw_imu_data[b, start_idx:start_idx+self.output_length]
#         #     selected_imu_data.append(selected)
#         # selected_imu_data = torch.stack(selected_imu_data, dim=0)  # [batch_size, output_length, 6]
#         # print(selected_imu_data)
#         # raise
#         return selected_imu_data
#         # return {
#         #     'sampled_imu_data': selected_imu_data,
#         #     'start_indices': start_indices,
#         #     'max_similarities': max_similarities,
#         #     'similarities': similarities
#         # }

# class GuidedIMUSimilaritySampler(IMUSimilaritySampler):
#     def __init__(self, shared_attn, imu_encoder, imu_dim=256, visual_dim=512, embed_dim=512,
#                  window_size=11, input_length=111, output_length=101):
#         super().__init__(imu_encoder, imu_dim, visual_dim, embed_dim, window_size, input_length, output_length)
#         self.atten = shared_attn
#
#     def compute_similarity(self, imu_window, visual_feature, pos_embedding):
#         """
#         注意力引导的，计算IMU窗口与视觉特征的相似度
#
#         参数:
#             imu_window: [batch_size, seq_len*11, imu_dim]
#             visual_feature: [batch_size, seq_len, visual_dim]
#
#         返回:
#             similarity: 相似度得分 [batch_size]
#         """
#         # 嵌入IMU窗口和视觉特征
#         seq_len = visual_feature.shape[1]  # 1
#         imu_window = torch.cat([imu_window[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
#         imu_window = self.imu_encoder(imu_window)
#         imu_embedded = self.imu_embedder(imu_window)  # [batch_size, seq_len, embed_dim]
#         visual_embedded = self.visual_embedder(visual_feature)  # [batch_size, embed_dim]
#
#         # 计算视觉特征与IMU窗口各点的相似度
#         visual_embedded_expanded = visual_embedded  # [batch_size, 1, embed_dim]
#
#         visual_embedded_expanded += pos_embedding
#         imu_embedded += pos_embedding
#
#         # 视觉引导的imu
#         imu_embedded = self.atten(visual_embedded_expanded, imu_embedded, imu_embedded)
#         # visual_embedded_expanded = self.atten(imu_embedded, visual_embedded_expanded, visual_embedded_expanded)
#         similarity = F.cosine_similarity(imu_embedded.squeeze(1), visual_embedded_expanded.squeeze(1), dim=1)
#         # similarity = torch.bmm(
#         #     visual_embedded_expanded,
#         #     imu_embedded.permute(0, 2, 1)
#         # ).squeeze(-1).squeeze(-1)  # [batch_size]
#
#         # 平均相似度得分
#         # similarity = sim_matrix.mean(dim=1)  # [batch_size]
#         return similarity
#
#     def forward(self, raw_imu_data, visual_feature, pos_embedding):
#         """
#         参数:
#             raw_imu_data: [batch_size, input_length, imu_dim]
#             visual_features: [batch_size, seq_len, visual_dim]
#             pos_embedding: [1, 1, embed_dim]
#         """
#         batch_size = raw_imu_data.shape[0]
#
#         # 获取第一个视觉特征 (假设它对应时间序列的开始)
#         #visual_feature = visual_feature[:, :1]  # [batch_size, 1, visual_dim]
#
#         # 1. 计算所有可能窗口的相似度
#         similarities = []
#         for i in range(self.max_start_idx + 1):  # 0到10
#             # 提取IMU窗口
#             imu_window = raw_imu_data[:, i:i+self.window_size]  # [batch_size, window_size, 6]
#
#             # 计算相似度
#             sim = self.compute_similarity(imu_window, visual_feature, pos_embedding)
#             similarities.append(sim.unsqueeze(1))  # [batch_size, 1]
#         similarities = torch.cat(similarities, dim=1)  # [batch_size, 11]
#
#         # 2. 选择相似度最大的起始位置
#         max_similarities, start_indices = torch.max(similarities, dim=1)  # [batch_size]
#         # start_indices = torch.zeros_like(start_indices) # 对比
#         # start_indices = torch.full_like(start_indices, 10)
#         # print(start_indices)
#
#         # 3. 根据起始位置提取连续的110个IMU数据点
#         selected_imu_data = []
#         for b in range(batch_size):
#             start_idx = start_indices[b].item()
#             # print(start_idx)
#             selected = raw_imu_data[b, start_idx:start_idx+self.output_length]
#             selected_imu_data.append(selected)
#         selected_imu_data = torch.stack(selected_imu_data, dim=0)  # [batch_size, output_length, 6]
#         return selected_imu_data
        

# class WrapperModel(torch.nn.Module):
#     def __init__(self, params):
#         super(WrapperModel, self).__init__()
#         self.inertial_encoder = Inertial_encoder(params)
#     def forward(self, imus):
#         feat_i = self.inertial_encoder(imus)
#         return feat_i
# class PoseTransformerEncoder_imusampler(PoseTemporalTransformerEncoder):
#     def __init__(self, params):
#         super().__init__(params)
#         self.params = params
#         self.Feature_net = WrapperModel(params)
#         self.imu_sampler = IMUSimilaritySampler(self.Feature_net, self.params.i_f_len, self.params.v_f_len, self.params.embed_dim, temperature = self.params.temperature)
#
#     def forward(self, visual_inertial_gps_features,  raw_imu_data):
#         seq_len = visual_inertial_gps_features.shape[1]
#         imu_data = self.imu_sampler(raw_imu_data, visual_inertial_gps_features[:, :1, :self.params.v_f_len])
#         imu_data = torch.cat([imu_data[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
#         imu_data = self.Feature_net(imu_data)
#         v_i_feature = torch.cat((visual_inertial_gps_features[:, :, :self.params.v_f_len], imu_data), dim=-1)
#
#         output = super().forward((v_i_feature, None, None), None)
#         return output
#
# class PoseTransformerEncoder_Guidedimusampler(PoseTemporalTransformerEncoder):
#     def __init__(self, params):
#         super().__init__(params)
#         self.params = params
#         self.Feature_net = WrapperModel(params)
#         self.imu_sampler = GuidedIMUSimilaritySampler(self.transformer_encoder.layers[0].attn, self.Feature_net, self.params.i_f_len, self.params.v_f_len, self.params.embed_dim)
#
#     def forward(self, visual_inertial_gps_features,  raw_imu_data):
#         seq_len = visual_inertial_gps_features.shape[1]
#         pos_embedding = self.positional_embedding(seq_len).to(visual_inertial_gps_features.device)[:, 0, :]
#         imu_data = self.imu_sampler(raw_imu_data, visual_inertial_gps_features[:, :1, :self.params.v_f_len], pos_embedding)
#         imu_data = torch.cat([imu_data[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
#         imu_data = self.Feature_net(imu_data)
#         v_i_feature = torch.cat((visual_inertial_gps_features[:, :, :self.params.v_f_len], imu_data), dim=-1)
#
#         output = super().forward((v_i_feature, None, None), None)
#         return output
#
# class PoseTransformerEncoder_imupre(PoseTemporalTransformerEncoder):
#     def __init__(self, params):
#         super().__init__(params)
#         self.params = params
#         self.Feature_net = Preintegration_Inertial_encoder(params)
#
#     def forward(self, visual_inertial_features, imupreintegration, gps):
#         # 暂时每加gps
#         imu_feature = self.Feature_net(imupreintegration)
#         v_i_feature = torch.cat((visual_inertial_features[:, :, :self.params.v_f_len], imu_feature), dim=-1)
#
#         output = super().forward((v_i_feature, None, None), None)
#         return output
#
# class PoseTransformer_imupre(PoseTransformer):
#     def __init__(self, params):
#         super().__init__(params)
#         self.params = params
#         self.Feature_net = Preintegration_Inertial_encoder(params)
#
#     def forward(self, visual_inertial_features, imupreintegration, gps):
#         # 暂时每加gps
#         imu_feature = self.Feature_net(imupreintegration)
#         v_i_feature = torch.cat((visual_inertial_features[:, :, :self.params.v_f_len], imu_feature), dim=-1)
#
#         output = super().forward(v_i_feature, gps)
#         return output
#
# class PoseTransformer_test_imuencoder(PoseTransformer):
#     def __init__(self, params):
#         super().__init__(params)
#         self.params = params
#         self.Feature_net = WrapperModel(params)
#
#     def forward(self, visual_inertial_features, imu, gps):
#         seq_len = visual_inertial_features.shape[1]
#         imu_data = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
#         imu_feature = self.Feature_net(imu_data)
#         v_i_feature = torch.cat((visual_inertial_features[:, :, :self.params.v_f_len], imu_feature), dim=-1)
#
#         output = super().forward(v_i_feature, gps)
#         return output
#
# #### botvio component
# import src.models.components.botvio_component as botvio_component
#
# class PoseModel_Botvio(nn.Module):
#     def __init__(self, params, embed_dim=256, num_heads=8, num_layers=4, seq_len=10, batch_first=True):
#         super().__init__()
#         self.params = params
#         self.batch_first = params.batch_first
#         self.fusion_net = botvio_component.Trans_Fusion(dim=params.embed_dim, seq_len=params.seq_len)
#         self.pose_head = botvio_component.Pose_CNN(params.embed_dim)
#
#     def forward(self, batch, target): #记得删target
#         # 确保输入格式正确
#         visual_inertial_gps_features, _, _ = batch
#         if not self.batch_first:
#             visual_inertial_gps_features = visual_inertial_gps_features.transpose(0, 1)  # 转换为 (batch, seq_len, embed_dim)
#             seq_length = visual_inertial_gps_features.size(0)
#         else:
#             seq_length = visual_inertial_gps_features.size(1)
#
#         visual_feature = visual_inertial_gps_features[:, :, :self.params.v_f_len]
#         inertial_feature = visual_inertial_gps_features[:, :, self.params.v_f_len:]
#         fusion_feature = self.fusion_net(visual_feature, inertial_feature)
#         axisangle, translation = self.pose_head(fusion_feature)
#         output = torch.cat((axisangle, translation), dim=-1)
#         return output