import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from torch.utils.data import DataLoader

class DatasetDRNet(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """
    def __init__(self, n_channels=1, patch_size=120, sigma=25, isTrain=True, paths_H=None, t=10):
        super(DatasetDRNet, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.sigma = sigma
        self.t = t
        self.sigma_train = [5,70] # 5 - 70
        self.sigma_test = self.sigma
        self.isTrain = isTrain
        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        if paths_H == None:
            self.paths_H = util.get_image_paths(r"E:\Video Super-resolution\DataSet\DIV2K\HighResolution\DIV2K_train_HR\DIV2K_train_HR")
            # self.paths_H = util.get_image_paths(r"E:\image_denoising\datasets\DIV2Kpatch1")
        else:
            self.paths_H = util.get_image_paths(paths_H)

    def __getitem__(self, index):
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path
        if self.isTrain:
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            # step = random.randint(0,self.max_step)
            sigma_sum = 0 # justfy if the sigma is up to 70
            noises = []
            noise = torch.zeros_like(img_L)
            sigma_ = random.randint(self.sigma_train[0], self.sigma_train[1])
            if self.t == 1:
                noise += torch.randn(img_L.size()).mul_(sigma_/ 255.0)
                noises.append(noise)
                img_L.add_(noise)
            else:
                for i in range(self.t):
                    if i==0:
                        sigma = random.randint(0, sigma_//2)
                    elif i == (self.t-1):
                        sigma = sigma_-sigma_sum
                    else:
                        sigma = random.randint(0, (sigma_-sigma_sum)//2)

                    sigma_sum += sigma
                    # print(i, sigma_, sigma, sigma_sum)
                    # sigma_sum += sigma
                    # if sigma_sum > self.sigma_train[1]:
                    #     break
                    noise += torch.randn(img_L.size()).mul_(sigma/255.0)
                    noises.append(noise)
                img_L.add_(noise)
            # --------------------------------
            # clip
            # --------------------------------
            return img_L, img_H, noises
        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)
            h, w, c = img_L.shape
            # print(img_L.shape)
            if h % 8 != 0:
                img_L = img_L[:(h - h % 8), :, :]
                img_H = img_H[:(h - h % 8), :, :]
            if w % 8 != 0:
                img_L = img_L[:, :(w - w % 8),:]
                img_H = img_H[:, :(w - w % 8),:]
            # print(img_L.shape)
            # --------------------------------
            # add noise
            # --------------------------------
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_Ls = util.single2tensor3(img_L)
            img_Hs = util.single2tensor3(img_H)
            return img_Ls, img_Hs

    def __len__(self):
        return len(self.paths_H)

if __name__ == '__main__':
    train_dir = r'E:\image_denoising\aaaa\SIDD_patches\train'
    sigma = 30  # 训练时是5-70，测试时是30
    batch_size = 48
    patch_sizes = [128]
    in_channels = 3
    t = 1

    train_set = DatasetDRNet(in_channels,patch_sizes[0],sigma,True,t=t)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)

