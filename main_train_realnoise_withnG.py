import os
import torch
import torch.nn as nn
from dataloaders.data_rgb import get_training_data, get_validation_data

import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
import glob
import time
import scipy.io
from utils import utils_logger
from utils import utils_image as util
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from models.loss_ssim import SSIMLoss
from models.network_compdnet_4_real import Net
from visdom import Visdom

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
sigma = 'realnoise'
batch_size = 64
patch_sizes = 128
in_channels = 3
hiddens = 64
lr = 0.00005
test_iter = 200
load_pretrained = True
logger_name = 'train'
t = 1

######### Set Seeds ###########
save_path =  os.path.join('model_zoo',str(sigma))
if not os.path.exists(save_path):
    os.makedirs(save_path)
utils_logger.logger_info(logger_name, os.path.join('model_zoo',str(sigma), logger_name + '.log'))
logger = logging.getLogger(logger_name)

seed = 100
# logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_dir = r'E:\image_denoising\SIDD_patches\train'

viz = Visdom()
viz.line([0.0], [0.], win='PSNR', opts=dict(title='Train PSNR'))
viz.line([0.0], [0.], win='Loss', opts=dict(title='Train Loss'))


def test(model):
    model.eval()
    avg_psnr = 0
    count = 0
    torch.manual_seed(0)
    i_imgs, i_blocks, _, _, _ = all_noisy_imgs.shape
    psnrs = []
    ssims = []
    import utils.utils_image as util
    for i_img in range(i_imgs):
        for i_block in range(i_blocks):
            noise = transforms.ToTensor()(Image.fromarray(all_noisy_imgs[i_img][i_block])).unsqueeze(0)
            noise = noise.cuda()
            with torch.no_grad():
                pred = compnet(x=noise, t=t, train=False)
            pred = pred.detach().float().cpu()
            gt = transforms.ToTensor()((Image.fromarray(all_clean_imgs[i_img][i_block])))
            gt = gt.unsqueeze(0)
            pred = util.tensor2uint(pred)
            gt = util.tensor2uint(gt)
            psnr_t = util.calculate_psnr(pred, gt)
            ssim_t = util.calculate_ssim(pred, gt)
            psnrs.append(psnr_t)
            ssims.append(ssim_t)
            avg_psnr += psnr_t
            count += 1
    avg_psnr = avg_psnr / count
    return avg_psnr, psnrs

######### Model ###########
if __name__ == '__main__':

    compnet = Net(in_channels,in_channels,hiddens,t=t).cuda()
    compnet = torch.nn.DataParallel(compnet)
    if load_pretrained:
        compnet.load_state_dict(torch.load(os.path.join(save_path,'best.pth')))
    print('load success')

    ######### Scheduler ###########
    total_lossfn = nn.L1Loss().cuda()
    image_net_lossfn = SSIMLoss().cuda()
    noise_net_lossfn = nn.MSELoss().cuda() 
    noise_net_lossfn2 = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(compnet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-7)

    ######### DataLoaders ###########
    img_options_train = {'patch_size': patch_sizes}
    train_dataset = get_training_data(train_dir, img_options_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              drop_last=False)
    best_psnr = 0
    best_epoch = 0
    best_iter = 0

    eval_now = 200
    print("Evaluation after every {" + str(eval_now) + "} Iterations !!!\n")
    all_noisy_imgs = scipy.io.loadmat(r'E:\image_denoising\testsets\ValidationNoisyBlocksSrgb.mat')[
    'ValidationNoisyBlocksSrgb']
    all_clean_imgs = scipy.io.loadmat(r'E:\image_denoising\testsets\ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']

    best = 0
    count = 0
    for epo in range(100000):

        for j, data in enumerate(train_loader):
            count += 1
            Ls, Hs = data

            Ls = Ls.cuda()
            Hs = Hs.cuda()

            y = compnet(Ls, 1)

            loss_image_net2 = total_lossfn(y, Hs)
            loss_noise_net = 0
            # loss_noise_net += noise_net_lossfn(torch.log(torch.softmax(noises_[0], 1)), torch.softmax(Ls-Hs, 1))
            loss = loss_image_net2
            viz.line([loss_image_net2.detach().cpu().numpy()], [count], win='Loss', update='append')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if count%test_iter == 0:
                print('now iter', count)

                test_val_, psnrs = test(compnet)
                # 画图
                viz.line([test_val_], [count], win='PSNR', update='append')

                if test_val_ > best:
                    torch.save(compnet.state_dict(), os.path.join(save_path,'best.pth'))
                    best = test_val_
                    logger.info('best PSNR:[{}] iter:[{}]'.format(test_val_, count))
                    # for i_psnr in range(len(psnrs)):
                    #     logger.info('{:->4d}| {:<4.2f}dB'.format(i_psnr, psnrs[i_psnr]))
                    # logger.info('---------------')
                compnet.train()
        # torch.save(compnet.state_dict(), os.path.join(save_path, 'now.pth'))

        if epo % 10 == 0 and epo > 0:
            scheduler.step()
            print('epoch', epo, ' current learning rate', optimizer.param_groups[0]['lr'])

