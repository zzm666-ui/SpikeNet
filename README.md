## SpikeNet
[ICML 2026] SpikeNet: Sparse Spike-Driven Mask Vector Transformer for Energy-Efficient and Stable Spiking Point Cloud Processing

Zhiming Zhou, Yong He, Chaoxu Mu, Qiaoyun Wu, Ajmal Saeed Mian

## Install

```bash
conda create -n SpikeNet python=3.7 -y
conda activate SpikeNet
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=10.2 -c pytorch -y
# if you are using Ampere GPUs (e.g., A100 and 30X0), please install compatible Pytorch and CUDA versions, like:
# pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2
pip install spikingjelly
pip install pointnet2_ops_lib/.
```

## Useage

### Classification ModelNet40
**Train**: The dataset will be automatically downloaded, run following command to train.

By default, it will create a folder named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.
```bash
cd classification_ModelNet40
# train 
python main.py 
```

### Classification ScanObjectNN

The dataset will be automatically downloaded

- Train 
```bash
cd classification_ScanObjectNN
# train 
python main.py 
```
By default, it will create a fold named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.


### Part segmentation

- Make data folder and download the dataset
```bash
cd part_segmentation
cd data
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

- Train 
```bash
# train 
python main.py 
```

## Citation

If you find SpikeNet useful to your research, please cite our work as an acknowledgment.

```bash
@inproceedings{
zhou2026spikenet,
title={SpikeNet: Sparse Spike-Driven Mask Vector Transformer for Energy-Efficient and Stable Spiking Point Cloud Processing},
author={Zhiming Zhou and Yong He and Chaoxu Mu and Qiaoyun Wu and Ajmal Saeed Mian},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=7BpcmBjQL0}
}
```





