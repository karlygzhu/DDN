# DDN
Datasets:
----------
## Training dataset for AWGN denoising
- DIV2K
## Test datasets for AWGN denoising
- CBSD68
- Kodak24
## Training dataset for real-world image denoising
- SIDD
## Test dataset for real-world image denoising
- DND

## Environments
----------
- PyTorch 1.6.1
- CUDA 10.2
- Python 3.7

Directory structure
----------
- models (network, train method definition)
- model_zoo (pre-trained models, download in:
Link: https://pan.baidu.com/s/1mPRVyCmKFIZcEzSGYMPXPw  
Code: jxba
--baidu device)
- testsets
- utils

# Commands
----------
## Training
## Train DDN for AWGN denoising
### dataset_compnet.py --self.paths_H [path of training set]
### run main_train_compnet.py --t [can be set as 1, 3 or 5]
## Train DDN for real-world image denoising
### run main_train_realnoise.py --train_dir [path of training set] --all_noisy_imgs [path of validation set] --all_clean_imgs [path of validation set]

## Test
## Test DDN for AWGN denoising
### Copying test color noisy images into DDN/testsets
### run main_test_compnet.py --t [can be set as 1, 3 or 5] --datasetname
## Test DDN for real-world image denoising
### run main_test_realnoise.py --train_dir [path of training set] --all_noisy_imgs [path of validation set] --all_clean_imgs [path of validation set]
