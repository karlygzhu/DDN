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
from torchnet.logger import VisdomPlotLogger, VisdomLogger
# from visdom import Visdom

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'

sigma = 30 # 训练时是5-70，测试时是30
batch_size = 8
patch_sizes = [128]
in_channels = 3

hiddens = 64
lr = 0.00005
test_iter = 100
load_pretrained = False
logger_name = 'train'
model_name = 'v4'
t = 3
color_type = 'color'
save_path =  os.path.join('model_zoo',str(sigma),str(t),color_type,model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
utils_logger.logger_info(logger_name, os.path.join('model_zoo', str(sigma), str(t), color_type,model_name, logger_name + '.log'))
logger = logging.getLogger(logger_name)

# viz = Visdom()

seed = 100
# logger.info('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def test(model):
    model.eval()
    avg_psnr = 0
    psnrs = []
    count = 0
    for i, data in enumerate(test_loader):
        count += 1
        L, H = data
        L = L.cuda()
        H = H.cuda()
        with torch.no_grad():
            out, *_ = compnet(x=L,t=t,train = False)
            out = util.tensor2uint(out)
        H = util.tensor2uint(H)
        current_psnr = util.calculate_psnr(out, H, border=0)
        psnrs.append(current_psnr)
        avg_psnr += current_psnr
    avg_psnr = avg_psnr / count
    return avg_psnr, psnrs

if __name__ == '__main__':
    # total_loss_logger = VisdomPlotLogger('line', opts={'title': 'Loss'})
    # image_net_logger =  VisdomPlotLogger('line', opts={'title': 'Image net Loss'})
    # noise_net_logger = VisdomPlotLogger('line', opts={'title': 'Noise net Loss'})
    # psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})

    train_set = DatasetDRNet(in_channels,patch_sizes[0],sigma,True,t=t)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)
    test_set = DatasetDRNet(in_channels,patch_sizes[0],sigma,False,r'testsets\set5',t=t)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=0,
                             drop_last=False, pin_memory=True)

    compnet = Net(in_channels,in_channels,hiddens,t=t).cuda()
    # compnet = nc.init_weights(compnet)
    compnet = torch.nn.DataParallel(compnet)
    if load_pretrained:
        compnet.load_state_dict(torch.load(os.path.join(save_path,'best.pth')))

    # loss function
    total_lossfn = nn.L1Loss().cuda()
    image_net_lossfn = SSIMLoss().cuda()
    # image_net_lossfn = nn.MSELoss().cuda()
    noise_net_lossfn = nn.KLDivLoss().cuda() 
    noise_net_lossfn2 = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(compnet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-7)

    best = 0
    count = 0
    count_loss = 0
    for epo in range(100000):
        for j, data in enumerate(train_loader):
            count += 1
            count_loss += 1
            # print(i)
            Ls, Hs, noises = data

            Ls = Ls.cuda()
            Hs = Hs.cuda()
            for i in range(len(noises)):
                noises[i] = noises[i].cuda()

            y, noises_ = compnet(Ls, len(noises))
            # loss_image_net1 = image_net_lossfn(out, Hs)
            loss_image_net2 = total_lossfn(y, Hs)
            loss_noise_net = 0
            for i in range(1, len(noises)):
                loss_noise_net += noise_net_lossfn(torch.log(torch.softmax(noises_[i],1)), torch.softmax(noises[i],1))
            # print(loss_noise_net)
            loss = loss_image_net2 + loss_noise_net
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if count%test_iter == 0:
                print('now iter', count)
                test_val_, psnrs = test(compnet)
                if test_val_ > best:
                    torch.save(compnet.state_dict(), os.path.join(save_path,'best.pth'))
                    best = test_val_
                    print('now_best',best)
                    logger.info('best PSNR:[{}] iter:[{}]'.format(test_val_, count))
                    # for i_psnr in range(len(psnrs)):
                    #     logger.info('{:->4d}| {:<4.2f}dB'.format(i_psnr, psnrs[i_psnr]))
                    logger.info('---------------')
                # if count == 1:
                #     viz.line([test_val_], [count], win='PSNR', opts=dict(title='Train PSNR'))
                #     viz.line([loss.item() / count_loss], [count], win='Loss', opts=dict(title='Train Loss'))
                # else:
                #     viz.line([test_val_], [count], win='PSNR', update='append')
                #     viz.line([loss.item() / count_loss], [count], win='Loss', update='append')
                compnet.train()
        # torch.save(compnet.state_dict(), os.path.join(save_path, 'now.pth'))
        # if epo % 10 == 0 and epo > 0:
        #     index_patch = np.random.choice(len(patch_sizes), 1)
        #     train_set = DatasetDRNet(in_channels, patch_sizes[int(index_patch)], sigma, True)
        #     train_loader = DataLoader(train_set,
        #                               batch_size=batch_size,
        #                               shuffle=True,
        #                               num_workers=0,
        #                               drop_last=True,
        #                               pin_memory=True)

        if epo % 50 == 0 and epo > 0:
            scheduler.step()
            print('epoch', epo, ' current learning rate', optimizer.param_groups[0]['lr'])

