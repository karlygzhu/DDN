import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from data.dataset_compnet import DatasetDRNet

from models.network_compdnet_4 import Net

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
import torch
import torch.nn as nn
from models.loss_ssim import SSIMLoss
from torch.optim import lr_scheduler
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


sigma = [10,30,50]
netname = 'color'
datasetname = ['cbsd68']
batch_size = 1
patch_sizes = [128]
in_channels = 3
hiddens = 64
t = 3
model_name = 'v4'


lr = 0.0001
test_iter = 200
load_pretrained = True
logger_name = 'test'

save_path =  os.path.join('model_zoo',str(30),str(t),netname,model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
utils_logger.logger_info(logger_name, os.path.join('model_zoo',str(30),str(t),netname,model_name, logger_name + '.log'))
logger = logging.getLogger(logger_name)

seed = 100
# logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def test(model):
    model.eval()
    avg_psnr = 0
    avg_ssim = 0
    psnrs = []
    ssim = []
    count = 0
    for i, data in enumerate(test_loader):
        count += 1
        L, H = data
        L = L.cuda()
        H = H.cuda()
        with torch.no_grad():
            out,_ = compnet(x=L,t=t,train = False)
            out = util.tensor2uint(out)
        H = util.tensor2uint(H)
        current_psnr = util.calculate_psnr(out, H, border=0)
        current_ssim = util.calculate_ssim(out, H, border=0)

        psnrs.append(current_psnr)
        ssim.append(current_ssim)

        avg_psnr += current_psnr
        avg_ssim += current_ssim

    avg_psnr = avg_psnr / count
    avg_ssim = avg_ssim / count

    return avg_psnr,avg_ssim,psnrs

if __name__ == '__main__':
    for datasetn in datasetname:
        for sigma_ in sigma:
            test_set = DatasetDRNet(in_channels,patch_sizes[0],sigma_,False,'testsets\\'+datasetn)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=0,
                                     drop_last=False, pin_memory=True)

            compnet = Net(in_channels,in_channels,hiddens,t=t).cuda()
            compnet = torch.nn.DataParallel(compnet)
            if load_pretrained:
                compnet.load_state_dict(torch.load(os.path.join(save_path,'best.pth')))

            best = 0
            count = 3200
            test_val_, test_ssim, psnrs = test(compnet)
            logger.info('dataset:[{}] sigma:[{}] test PSNR:[{}] SSIM:[{}] iter:[{}]'.format(datasetn, sigma_, test_val_,test_ssim, count))
        # for i_psnr in range(len(psnrs)):
        #     logger.info('{:->4d}| {:<4.2f}dB'.format(i_psnr, psnrs[i_psnr]))
