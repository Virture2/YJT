import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from path import Path
from src.utils.kitti_utils import rotationError, read_pose_from_text
from src.utils import new_custom_transform
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d

IMU_FREQ = 10

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

class KITTI(Dataset):
    def __init__(self, root,
                 sequence_length=11,
                 train_seqs=['00', '01', '02', '04', '06', '09'],
                 transform=None):
        
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.transform = transform
        self.train_seqs = train_seqs
        self.make_dataset()

    def make_dataset(self):
        sequence_set = []
        for folder in self.train_seqs:
            gnss = read_gnss_from_text(self.root/'oxts'/folder/'{}.txt'.format(folder))
            poses, poses_rel = read_pose_from_text(self.root/'poses/{}.txt'.format(folder))
            imus = sio.loadmat(self.root/'imus/{}.mat'.format(folder))['imu_data_interp']
            fpaths = sorted((self.root/'sequences/{}/image_2'.format(folder)).files("*.png"))
            # for i in range(0, len(fpaths) - self.sequence_length, self.sequence_length):
            #     img_samples = fpaths[i:i + self.sequence_length]
            #     imu_samples = imus[i * IMU_FREQ:(i + self.sequence_length - 1) * IMU_FREQ + 1]
            #     pose_samples = poses[i:i + self.sequence_length]
            #     pose_rel_samples = poses_rel[i:i + self.sequence_length - 1]
            #
            #     # GNSS 差分
            #     gnss_segment = gnss[i:i + self.sequence_length]
            #     gnss_samples = gnss_segment[1:] - gnss_segment[:-1]
            #
            #     # 计算旋转误差
            #     segment_rot = rotationError(pose_samples[0], pose_samples[-1])
            #
            #     sample = {
            #         'imgs': img_samples,
            #         'imus': imu_samples,
            #         'gts': pose_rel_samples,
            #         'rot': segment_rot,
            #         'gnss': gnss_samples
            #     }
            #     sequence_set.append(sample)
            for i in range(len(fpaths)-self.sequence_length):
                img_samples = fpaths[i:i+self.sequence_length]
                imu_samples = imus[i*IMU_FREQ:(i+self.sequence_length-1)*IMU_FREQ+1]
                pose_samples = poses[i:i+self.sequence_length]
                pose_rel_samples = poses_rel[i:i+self.sequence_length-1]
                # gnss_samples = gnss[i:i+self.sequence_length]
                gnss_samples = gnss[i + 1:i + self.sequence_length] - gnss[i:i + self.sequence_length - 1]
                # print(f"[DEBUG] gnss_samples shape at index {i}: {gnss_samples.shape}")
                segment_rot = rotationError(pose_samples[0], pose_samples[-1])
                sample = {'imgs':img_samples, 'imus':imu_samples, 'gts': pose_rel_samples, 'rot': segment_rot, 'gnss': gnss_samples}
                sequence_set.append(sample)
        self.samples = sequence_set
        
        # Generate weights based on the rotation of the training segments
        # Weights are calculated based on the histogram of rotations according to the method in https://github.com/YyzHarry/imbalanced-regression
        rot_list = np.array([np.cbrt(item['rot']*180/np.pi) for item in self.samples])
        rot_range = np.linspace(np.min(rot_list), np.max(rot_list), num=10)
        indexes = np.digitize(rot_list, rot_range, right=False)
        num_samples_of_bins = dict(Counter(indexes))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(1, len(rot_range)+1)]

        # Apply 1d convolution to get the smoothed effective label distribution
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=7, sigma=5)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

        self.weights = [np.float32(1/eff_label_dist[bin_idx-1]) for bin_idx in indexes]

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [np.asarray(Image.open(img)) for img in sample['imgs']]
        
        if self.transform is not None:
            imgs, imus, gts, gnss = self.transform(imgs, np.copy(sample['imus']), np.copy(sample['gts']), np.copy(sample['gnss']))
        else:
            imus = np.copy(sample['imus'])
            gts = np.copy(sample['gts']).astype(np.float32)
            gnss = np.copy(sample['gnss']).astype(np.float32)
        
        rot = sample['rot'].astype(np.float32)
        weight = self.weights[index]

        return (imgs.to(torch.float), torch.from_numpy(imus).to(torch.float), torch.from_numpy(gnss).to(torch.float), rot, weight), torch.from_numpy(gts).to(torch.float)

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Training sequences: '
        for seq in self.train_seqs:
            fmt_str += '{} '.format(seq)
        fmt_str += '\n'
        fmt_str += '    Number of segments: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window



