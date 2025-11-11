______________________________________________________________________
## Installation

Make sure you installed correct PyTorch version for your specific development environment.
Other requirements can be installed via `pip`.

#### Pip

```bash
# clone project
git clone -b master https://github.com/Virture2/YJT.git
cd YJT

# [OPTIONAL] create conda environment
conda create -n YJT python=3.9
conda activate YJT

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Downloading KITTI Dataset

Files shared via cloud storage：sequences
Link: https://pan.baidu.com/s/1JIR5LyBAFIZmYa6cggclww Extraction code: gvfn 
Place the sequences folder under the data/kitti_data/ directory.

## Download pre-trained weights

Link: https://pan.baidu.com/s/1_DzkG3QQR7aHPGmgNKVCGQ Extraction code: 7pcc 
Create a folder named `pretrained_models` under the YJT directory and place the three pretrained weights into this folder.

### Caching Latents for KITTI Dataset  

```bash
cd data
python latent_caching.py
python latent_val_caching.py
```

## Replicating Experiments in Paper

After saving latents for KITTI dataset, you can run following command to run the experiments.

```bash
python src/train.py experiment=new_latent_kitti_vio_weighted_tf trainer=gpu logger=tensorboard  tags=['TE, 11, L1, 40, no, yes']
```


