# DDN
## Abstract
----------
Image denoising is a significant task in computer vision. Previous studies have mostly concentrated on removing noise with specific levels. The blind image denoising approach has recently gained more popularity due to its adaptability. Nonetheless, existing deep learning methods only train networks to learn the direct projection from noisy images to clean ones, which limits their denoising performance. This paper proposes a novel perspective for blind denoising by converting the static image denoising problem into a dynamic process inspired by the diffusion model. To achieve this, we present a new method that views a noisy image as a mid-state of a Gaussian diffusion process. Specifically, the image noise is separated into multiple sub-level noises through the diffusion process, and sub-sequently eliminated in a sequential manner. Furthermore, we propose a diffusion denoising network (DDN) that comprises a Feature Extraction Module (FEM) for extracting image features and a Diffusion Noise Estimation Module (DNEM) for estimating the sub-level noises. Our experiments demonstrate that our proposed method outperforms existing methods and achieves state-of-the-art results in blind additive white Gaussian noise (AWGN) and real-world image denoising.

Datasets:
----------
Training dataset for AWGN denoising
- DIV2K

Test datasets for AWGN denoising
- CBSD68
- Kodak24

Training dataset for real-world image denoising
- SIDD
  
Test dataset for real-world image denoising
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
Train DDN for AWGN denoising:
dataset_compnet.py --self.paths_H [path of training set]
run main_train_compnet.py --t [can be set as 1, 3 or 5]

Train DDN for real-world image denoising:
run main_train_realnoise_withnG.py --train_dir [path of training set] --all_noisy_imgs [path of validation set] --all_clean_imgs [path of validation set]

## Test
Test DDN for AWGN denoising:
Copying test color noisy images into DDN/testsets
run main_test_compnet.py --t [can be set as 1, 3 or 5] --datasetname

Test DDN for real-world image denoising:
run main_test_realnoise.py  --all_noisy_imgs [path of test set] --all_clean_imgs [path of test set]
