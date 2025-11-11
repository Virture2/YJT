# import dependencies
import os
from tqdm import tqdm
import numpy as np

import glob
import scipy.io as sio
import torch
from src.utils.kitti_utils import read_pose_from_text, saveSequence
# from src.utils.kitti_eval import plotPath_2D, kitti_eval, data_partition
from src.utils.new_kitti_eval import plotPath_2D, kitti_eval, data_partition
from src.models.components.new_vsvio import Encoder
# from src.models.components.vsvio import Encoder

from natsort import natsorted


# wrap the model with an encoder for testing
class WrapperModel(torch.nn.Module):
    def __init__(self, params):
        super(WrapperModel, self).__init__()
        self.Feature_net = Encoder(params)
    # def forward(self, imgs, imus):
    #     feat_v, feat_i = self.Feature_net(imgs, imus)
    #     memory = torch.cat((feat_v, feat_i), 2)
    #     return memory
    def forward(self, imgs, imus, gnss):
        feat_v, feat_i, feat_g = self.Feature_net(imgs, imus, gnss)
        memory = torch.cat((feat_v, feat_i, feat_g), 2)
        return memory


class KITTI_tester_latent():
    def __init__(self, args, wrapper_weights_path, gnss_weights_path, use_history_in_eval=False):
    # def __init__(self, args, wrapper_weights_path, use_history_in_eval=False):
        super(KITTI_tester_latent, self).__init__()

        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))
        self.args = args

        # Initialize and load pretrained weights for the wrapper model
        self.wrapper_model = WrapperModel(args)
        self.load_wrapper_weights(wrapper_weights_path,gnss_weights_path)
        # self.load_wrapper_weights(wrapper_weights_path)
        self.wrapper_model.eval()
        self.wrapper_model.to(self.args.device)
        self.use_history_in_eval = use_history_in_eval

    def load_wrapper_weights(self, wrapper_weights_path, gnss_weights_path):
        model_dict = self.wrapper_model.state_dict()

        # === 第一次加载视觉+IMU预训练权重 ===
        if os.path.exists(wrapper_weights_path):
            pretrained_vio = torch.load(wrapper_weights_path, map_location='cpu')
            update_dict_vio = {k: v for k, v in pretrained_vio.items() if k in model_dict}
            print(f"[视觉+IMU] 加载了 {len(update_dict_vio)} 个权重")
            model_dict.update(update_dict_vio)
        else:
            print(f"[WARNING] 视觉+IMU权重文件未找到: {wrapper_weights_path}")

        # === 再加载 GNSS 分支预训练权重（只更新 GNSS 分支）===
        if os.path.exists(gnss_weights_path):
            pretrained_gnss = torch.load(gnss_weights_path, map_location='cpu')
            gnss_update = {k: v for k, v in pretrained_gnss.items() if "gnss_encoder" in k}
            # gnss_update = {
            #     "Feature_net." + k: v
            #     for k, v in pretrained_gnss.items()
            #     if "gnss_encoder" in k
            # }
            print(f"[GNSS] 加载了 {len(gnss_update)} 个GNSS分支权重")
            model_dict.update(gnss_update)
        elif gnss_weights_path:
            print(f"[WARNING] GNSS权重文件未找到: {gnss_weights_path}")

        # === 最终统一加载所有更新的权重 ===
        self.wrapper_model.load_state_dict(model_dict)
    #     print(f"[总计] 成功加载的参数数量: {len(update_dict_vio) + len(gnss_update)}")

    # def load_wrapper_weights(self, weights_path):
    #     if os.path.exists(weights_path):
    #         pretrained_w = torch.load(weights_path, map_location='cpu')
    #
    #         model_dict = self.wrapper_model.state_dict()
    #         update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
    #
    #         # Check if update dict is equal to model dict
    #         # assert len(update_dict.keys()) == len(self.wrapper_model.Feature_net.state_dict().keys()), "Some weights are not loaded"
    #
    #         if len(update_dict.keys()) != len(model_dict.keys()):
    #             print(
    #                 f"[Warning] Some weights are not loaded.\nLoaded {len(update_dict)} keys out of {len(model_dict)}.\n"
    #                 "This is normal if you added new branches like GNSS."
    #             )
    #         self.wrapper_model.load_state_dict(update_dict, strict=False)
    #
    #         # self.wrapper_model.load_state_dict(update_dict)
    #
    #         print(f"Loaded wrapper model weights from {weights_path}")
    #     else:
    #         print(f"Warning: Wrapper model weights not found at {weights_path}")
    # def load_wrapper_weights(self, weights_path):
    #     import pprint
    #
    #     if os.path.exists(weights_path):
    #         print(f"[INFO] Loading weights from {weights_path}")
    #         pretrained_w = torch.load(weights_path, map_location='cpu')
    #
    #         print(f"\n====== [DEBUG] Pretrained weights keys: ({len(pretrained_w.keys())} keys) ======")
    #         pprint.pprint(list(pretrained_w.keys()))
    #         print("============================================================\n")
    #
    #         model_dict = self.wrapper_model.state_dict()
    #         print(f"\n====== [DEBUG] Model expected keys: ({len(model_dict.keys())} keys) ======")
    #         pprint.pprint(list(model_dict.keys()))
    #         print("============================================================\n")
    #
    #         # Check if the shapes match (useful debug info)
    #         for k in pretrained_w.keys():
    #             if k in model_dict:
    #                 if pretrained_w[k].shape != model_dict[k].shape:
    #                     print(f"[WARNING] Shape mismatch: {k}: {pretrained_w[k].shape} vs {model_dict[k].shape}")
    #
    #         update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict and v.shape == model_dict[k].shape}
    #
    #         # Check if update dict is equal to model dict
    #         assert len(update_dict.keys()) == len(model_dict.keys()), (
    #             f"Some weights are not loaded.\n"
    #             f"Loaded {len(update_dict.keys())} keys out of {len(model_dict.keys())}.\n"
    #             f"Consider partial loading or adjusting your model."
    #         )
    #
    #         self.wrapper_model.load_state_dict(update_dict)
    #         print(f"[INFO] Loaded wrapper model weights successfully.")
    #     else:
    #         print(f"[WARNING] Wrapper model weights not found at {weights_path}")

    def test_one_path(self, net, df, num_gpu=1):
        pose_list = []
        self.hist = None
        for i, (image_seq, imu_seq, gnss_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
        # for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
            x_in = image_seq.unsqueeze(0).repeat(num_gpu,1,1,1,1).to(self.args.device)
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu,1,1).to(self.args.device)
            ################
            g_in = gnss_seq.unsqueeze(0).repeat(num_gpu, 1, 1).to(self.args.device)
            ###############
            with torch.inference_mode():
                # Generate latent representations
                # latents = self.wrapper_model(x_in, i_in)
                latents = self.wrapper_model(x_in, i_in, g_in)
                # accumulate poses by passing latents to the main model

                if (self.hist is not None) and self.use_history_in_eval:
                    results = torch.zeros(latents.shape[0], latents.shape[1], 6)
                    for idx in range(latents.shape[1]):
                        self.hist = torch.roll(self.hist, -1, 1) # shift so that index 0 becomes last one, shift in seq dim
                        self.hist[:,-1,:] = latents[:,idx,:]
                        x = (self.hist, None, None)
                        result = net(x, gt_seq) # batch_size, seq_len, 6
                        results[:,idx,:] = result[:,-1,:]
                    pose = results
                else:
                    self.hist = latents
                    pose = net((latents, None, None), gt_seq)
                    print("latents shape:", latents.shape)
            pose_list.append(pose[0,:,:].detach().cpu().numpy())
        pose_est = np.vstack(pose_list)
        return pose_est

    def eval(self, net, selection=None, num_gpu=1, p=0.5):
        self.errors = []
        self.est = []
        for i, seq in enumerate(self.args.val_seq):
            print(f'testing sequence {seq}')
            pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)
            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = kitti_eval(pose_est, self.dataloader[i].poses_rel)

            self.est.append({'pose_est_global':pose_est_global, 'pose_gt_global':pose_gt_global, 'speed':speed})
            self.errors.append({'t_rel':t_rel, 'r_rel':r_rel, 't_rmse':t_rmse, 'r_rmse':r_rmse})

        return self.errors

    def generate_plots(self, save_dir, window_size):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D(seq,
                        self.est[i]['pose_gt_global'],
                        self.est[i]['pose_est_global'],
                        save_dir,
                        self.est[i]['speed'],
                        window_size)

    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir/'{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('Seq {} saved'.format(seq))


class KITTI_tester_latent_tokenized():
    def __init__(self, args, wrapper_weights_path, use_history_in_eval=False):
        super().__init__()

        # generate data loader for each path
        self.dataloader = []
        for seq in args.val_seq:
            self.dataloader.append(data_partition(args, seq))
        self.args = args

        # Initialize and load pretrained weights for the wrapper model
        self.wrapper_model = WrapperModel(args)
        self.load_wrapper_weights(wrapper_weights_path)
        self.wrapper_model.eval()
        self.wrapper_model.to(self.args.device)
        self.use_history_in_eval = use_history_in_eval

    def load_wrapper_weights(self, weights_path):
        if os.path.exists(weights_path):
            pretrained_w = torch.load(weights_path, map_location='cpu')

            model_dict = self.wrapper_model.state_dict()
            update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}

            # Check if update dict is equal to model dict
            assert len(update_dict.keys()) == len(self.wrapper_model.Feature_net.state_dict().keys()), "Some weights are not loaded"

            self.wrapper_model.load_state_dict(update_dict)
            print(f"Loaded wrapper model weights from {weights_path}")
        else:
            print(f"Warning: Wrapper model weights not found at {weights_path}")

    def test_one_path(self, net, df, num_gpu=1):
        pose_list = []
        self.hist = None
        for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
            x_in = image_seq.unsqueeze(0).repeat(num_gpu,1,1,1,1).to(self.args.device)
            i_in = imu_seq.unsqueeze(0).repeat(num_gpu,1,1).to(self.args.device)

            with torch.inference_mode():
                # Generate latent representations
                latents = self.wrapper_model(x_in, i_in)
                # accumulate poses by passing latents to the main model

                if (self.hist is not None) and self.use_history_in_eval:
                    results = torch.zeros(latents.shape[0], latents.shape[1], 6)
                    for idx in range(latents.shape[1]):
                        self.hist = torch.roll(self.hist, -1, 1) # shift so that index 0 becomes last one, shift in seq dim
                        self.hist[:,-1,:] = latents[:,idx,:]
                        x = (self.hist, None, None)
                        result, _ = net(x, gt_seq) # batch_size, seq_len, 6
                        results[:,idx,:] = result[:,-1,:]
                    pose = results
                else:
                    self.hist = latents
                    pose, _ = net((latents, None, None), gt_seq)
                    print("latents shape:", latents.shape)
            pose_list.append(pose[0,:,:].detach().cpu().numpy())
        pose_est = np.vstack(pose_list)
        return pose_est

    def eval(self, net, selection=None, num_gpu=1, p=0.5):
        self.errors = []
        self.est = []
        for i, seq in enumerate(self.args.val_seq):
            print(f'testing sequence {seq}')
            pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)
            pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = kitti_eval(pose_est, self.dataloader[i].poses_rel)

            self.est.append({'pose_est_global':pose_est_global, 'pose_gt_global':pose_gt_global, 'speed':speed})
            self.errors.append({'t_rel':t_rel, 'r_rel':r_rel, 't_rmse':t_rmse, 'r_rmse':r_rmse})

        return self.errors

    def generate_plots(self, save_dir, window_size):
        for i, seq in enumerate(self.args.val_seq):
            plotPath_2D(seq,
                        self.est[i]['pose_gt_global'],
                        self.est[i]['pose_est_global'],
                        save_dir,
                        self.est[i]['speed'],
                        window_size)

    def save_text(self, save_dir):
        for i, seq in enumerate(self.args.val_seq):
            path = save_dir/'{}_pred.txt'.format(seq)
            saveSequence(self.est[i]['pose_est_global'], path)
            print('Seq {} saved'.format(seq))

# # import dependencies
# import os
# from tqdm import tqdm
# import numpy as np
#
# import glob
# import scipy.io as sio
# import torch
# from src.utils.kitti_utils import read_pose_from_text, saveSequence
# from src.utils.kitti_eval import plotPath_2D, kitti_eval, data_partition
# from src.models.components.vsvio import Encoder
#
# from natsort import natsorted
#
#
# # wrap the model with an encoder for testing
# class WrapperModel(torch.nn.Module):
#     def __init__(self, params,raft_args):
#         super(WrapperModel, self).__init__()
#         self.Feature_net = Encoder(params,raft_args)
#
#     def forward(self, imgs, imus):
#         feat_v, feat_i = self.Feature_net(imgs, imus)
#         memory = torch.cat((feat_v, feat_i), 2)
#         return memory
#
#
# class KITTI_tester_latent():
#     def __init__(self, args,raft_args, wrapper_weights_path,use_history_in_eval=False):
#         super(KITTI_tester_latent, self).__init__()
#
#         # generate data loader for each path
#         self.dataloader = []
#         for seq in args.val_seq:
#             self.dataloader.append(data_partition(args, seq))
#         self.args = args
#
#         # Initialize and load pretrained weights for the wrapper model
#         self.wrapper_model = WrapperModel(args,raft_args)
#         self.load_wrapper_weights(wrapper_weights_path)
#         self.wrapper_model.eval()
#         self.wrapper_model.to(self.args.device)
#         self.use_history_in_eval = use_history_in_eval
#
#     def load_wrapper_weights(self, weights_path):
#         if os.path.exists(weights_path):
#             pretrained_w = torch.load(weights_path, map_location='cpu')
#
#             model_dict = self.wrapper_model.state_dict()
#             update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
#
#             # Check if update dict is equal to model dict
#             # assert len(update_dict.keys()) == len(
#             #     self.wrapper_model.Feature_net.state_dict().keys()), "Some weights are not loaded"
#
#             self.wrapper_model.load_state_dict(update_dict)
#             print(f"Loaded wrapper model weights from {weights_path}")
#         else:
#             print(f"Warning: Wrapper model weights not found at {weights_path}")
#
#     def test_one_path(self, net, df, num_gpu=1):
#         pose_list = []
#         self.hist = None
#         for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
#             x_in = image_seq.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1).to(self.args.device)
#             i_in = imu_seq.unsqueeze(0).repeat(num_gpu, 1, 1).to(self.args.device)
#
#             with torch.inference_mode():
#                 # Generate latent representations
#                 latents = self.wrapper_model(x_in, i_in)
#                 # accumulate poses by passing latents to the main model
#
#                 if (self.hist is not None) and self.use_history_in_eval:
#                     results = torch.zeros(latents.shape[0], latents.shape[1], 6)
#                     for idx in range(latents.shape[1]):
#                         self.hist = torch.roll(self.hist, -1,
#                                                1)  # shift so that index 0 becomes last one, shift in seq dim
#                         self.hist[:, -1, :] = latents[:, idx, :]
#                         x = (self.hist, None, None)
#                         result = net(x, gt_seq)  # batch_size, seq_len, 6
#                         results[:, idx, :] = result[:, -1, :]
#                     pose = results
#                 else:
#                     self.hist = latents
#                     pose = net((latents, None, None), gt_seq)
#             pose_list.append(pose[0, :, :].detach().cpu().numpy())
#         pose_est = np.vstack(pose_list)
#         return pose_est
#
#     def eval(self, net, selection=None, num_gpu=1, p=0.5):
#         self.errors = []
#         self.est = []
#         for i, seq in enumerate(self.args.val_seq):
#             print(f'testing sequence {seq}')
#             pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)
#             pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = kitti_eval(pose_est, self.dataloader[
#                 i].poses_rel)
#
#             self.est.append({'pose_est_global': pose_est_global, 'pose_gt_global': pose_gt_global, 'speed': speed})
#             self.errors.append({'t_rel': t_rel, 'r_rel': r_rel, 't_rmse': t_rmse, 'r_rmse': r_rmse})
#
#         return self.errors
#
#     def generate_plots(self, save_dir, window_size):
#         for i, seq in enumerate(self.args.val_seq):
#             plotPath_2D(seq,
#                         self.est[i]['pose_gt_global'],
#                         self.est[i]['pose_est_global'],
#                         save_dir,
#                         self.est[i]['speed'],
#                         window_size)
#
#     def save_text(self, save_dir):
#         for i, seq in enumerate(self.args.val_seq):
#             path = save_dir / '{}_pred.txt'.format(seq)
#             saveSequence(self.est[i]['pose_est_global'], path)
#             print('Seq {} saved'.format(seq))
#
#
# class KITTI_tester_latent_tokenized():
#     def __init__(self, args, wrapper_weights_path, use_history_in_eval=False):
#         super().__init__()
#
#         # generate data loader for each path
#         self.dataloader = []
#         for seq in args.val_seq:
#             self.dataloader.append(data_partition(args, seq))
#         self.args = args
#
#         # Initialize and load pretrained weights for the wrapper model
#         self.wrapper_model = WrapperModel(args)
#         self.load_wrapper_weights(wrapper_weights_path)
#         self.wrapper_model.eval()
#         self.wrapper_model.to(self.args.device)
#         self.use_history_in_eval = use_history_in_eval
#
#     def load_wrapper_weights(self, weights_path):
#         if os.path.exists(weights_path):
#             pretrained_w = torch.load(weights_path, map_location='cpu')
#
#             model_dict = self.wrapper_model.state_dict()
#             update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
#
#             # Check if update dict is equal to model dict
#             assert len(update_dict.keys()) == len(
#                 self.wrapper_model.Feature_net.state_dict().keys()), "Some weights are not loaded"
#
#             self.wrapper_model.load_state_dict(update_dict)
#             print(f"Loaded wrapper model weights from {weights_path}")
#         else:
#             print(f"Warning: Wrapper model weights not found at {weights_path}")
#
#     def test_one_path(self, net, df, num_gpu=1):
#         pose_list = []
#         self.hist = None
#         for i, (image_seq, imu_seq, gt_seq) in tqdm(enumerate(df), total=len(df), smoothing=0.9):
#             x_in = image_seq.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1).to(self.args.device)
#             i_in = imu_seq.unsqueeze(0).repeat(num_gpu, 1, 1).to(self.args.device)
#
#             with torch.inference_mode():
#                 # Generate latent representations
#                 latents = self.wrapper_model(x_in, i_in)
#                 # accumulate poses by passing latents to the main model
#
#                 if (self.hist is not None) and self.use_history_in_eval:
#                     results = torch.zeros(latents.shape[0], latents.shape[1], 6)
#                     for idx in range(latents.shape[1]):
#                         self.hist = torch.roll(self.hist, -1,
#                                                1)  # shift so that index 0 becomes last one, shift in seq dim
#                         self.hist[:, -1, :] = latents[:, idx, :]
#                         x = (self.hist, None, None)
#                         result, _ = net(x, gt_seq)  # batch_size, seq_len, 6
#                         results[:, idx, :] = result[:, -1, :]
#                     pose = results
#                 else:
#                     self.hist = latents
#                     pose, _ = net((latents, None, None), gt_seq)
#             pose_list.append(pose[0, :, :].detach().cpu().numpy())
#         pose_est = np.vstack(pose_list)
#         return pose_est
#
#     def eval(self, net, selection=None, num_gpu=1, p=0.5):
#         self.errors = []
#         self.est = []
#         for i, seq in enumerate(self.args.val_seq):
#             print(f'testing sequence {seq}')
#             pose_est = self.test_one_path(net, self.dataloader[i], num_gpu=num_gpu)
#             pose_est_global, pose_gt_global, t_rel, r_rel, t_rmse, r_rmse, speed = kitti_eval(pose_est, self.dataloader[
#                 i].poses_rel)
#
#             self.est.append({'pose_est_global': pose_est_global, 'pose_gt_global': pose_gt_global, 'speed': speed})
#             self.errors.append({'t_rel': t_rel, 'r_rel': r_rel, 't_rmse': t_rmse, 'r_rmse': r_rmse})
#
#         return self.errors
#
#     def generate_plots(self, save_dir, window_size):
#         for i, seq in enumerate(self.args.val_seq):
#             plotPath_2D(seq,
#                         self.est[i]['pose_gt_global'],
#                         self.est[i]['pose_est_global'],
#                         save_dir,
#                         self.est[i]['speed'],
#                         window_size)
#
#     def save_text(self, save_dir):
#         for i, seq in enumerate(self.args.val_seq):
#             path = save_dir / '{}_pred.txt'.format(seq)
#             saveSequence(self.est[i]['pose_est_global'], path)
#             print('Seq {} saved'.format(seq))




