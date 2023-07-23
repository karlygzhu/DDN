import torch
import torch.nn as nn
import models.basicblock as B
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 还是不好的话，可以去掉那两次注意力，把nc改成128
class D_Block(nn.Module):
    def __init__(self, channel_in, channel_out, deconv = False):
        super(D_Block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, stride=1,
                                padding=1)
        self.relu1 = nn.PReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3,
                                stride=1, padding=1)
        self.relu2 = nn.PReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=3, stride=1,
                                padding=1)
        self.relu3 = nn.PReLU()
        self.tail = B.conv(channel_in, channel_out, mode='CBR')
    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv_3(conc))
        out = torch.add(out, residual)
        out = self.tail(out)
        return out
class _down(nn.Module):
    def __init__(self, channel_in):
        super(_down, self).__init__()

        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=2*channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.maxpool(x)

        out = self.relu(self.conv(out))

        return out
class _up(nn.Module):
    def __init__(self, channel_in):
        super(_up, self).__init__()

        self.relu = nn.PReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv(x))

        out = self.subpixel(out)

        return out
class UNet_org(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(UNet_org, self).__init__()
        self.DCR_block11 = D_Block(in_nc, in_nc)
        self.DCR_block12 = D_Block(in_nc, in_nc)
        self.down1 = self.make_layer(_down, in_nc)
        self.DCR_block21 = D_Block(in_nc*2, in_nc*2)
        self.DCR_block22 = D_Block(in_nc*2, in_nc*2)
        self.down2 = self.make_layer(_down, in_nc*2)
        self.DCR_block31 = D_Block(in_nc*4, in_nc*4)
        self.DCR_block32 = D_Block(in_nc*4, in_nc*4)
        self.down3 = self.make_layer(_down, in_nc*4)
        self.DCR_block41 = D_Block(in_nc*8, in_nc*8)
        self.DCR_block42 = D_Block(in_nc*8, in_nc*8)
        self.up3 = self.make_layer(_up, in_nc*16)
        self.DCR_block33 = D_Block(in_nc*8, in_nc*8)
        self.DCR_block34 = D_Block(in_nc*8, in_nc*8)
        self.up2 = self.make_layer(_up, in_nc*8)
        self.DCR_block23 = D_Block(in_nc*4, in_nc*4)
        self.DCR_block24 = D_Block(in_nc*4, in_nc*4)
        self.up1 = self.make_layer(_up, in_nc*4)
        self.DCR_block13 = D_Block(in_nc*2, in_nc*2)
        self.DCR_block14 = D_Block(in_nc*2, out_nc)
        # self.conv_f = nn.Conv2d(in_channels=in_nc*2, out_channels=out_nc, kernel_size=1, stride=1, padding=0)
        # self.relu2 = nn.PReLU()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.DCR_block11(x)

        conc1 = self.DCR_block12(out)

        out = self.down1(conc1)

        out = self.DCR_block21(out)

        conc2 = self.DCR_block22(out)

        out = self.down2(conc2)

        out = self.DCR_block31(out)

        conc3 = self.DCR_block32(out)

        conc4 = self.down3(conc3)

        out = self.DCR_block41(conc4)

        out = self.DCR_block42(out)

        out = torch.cat([conc4, out], 1)

        out = self.up3(out)

        out = torch.cat([conc3, out], 1)

        out = self.DCR_block33(out)

        out = self.DCR_block34(out)

        out = self.up2(out)

        out = torch.cat([conc2, out], 1)

        out = self.DCR_block23(out)

        out = self.DCR_block24(out)

        out = self.up1(out)

        out = torch.cat([conc1, out], 1)

        out = self.DCR_block13(out)

        out = self.DCR_block14(out)

        # out = self.relu2(self.conv_f(out))

        return out
class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1
class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class UNet_att(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(UNet_att, self).__init__()
        self.DCR_block11 = D_Block(in_nc, in_nc)
        self.DCR_block12 = D_Block(in_nc, in_nc)
        self.down1 = self.make_layer(_down, in_nc)
        self.DCR_block21 = D_Block(in_nc*2, in_nc*2)
        self.DCR_block22 = D_Block(in_nc*2, in_nc*2)
        self.down2 = self.make_layer(_down, in_nc*2)
        self.DCR_block31 = D_Block(in_nc*4, in_nc*4)
        self.DCR_block32 = D_Block(in_nc*4, in_nc*4)
        self.down3 = self.make_layer(_down, in_nc*4)
        self.DCR_block41 = D_Block(in_nc*8, in_nc*8)
        self.DCR_block42 = D_Block(in_nc*8, in_nc*8)
        self.up3 = self.make_layer(_up, in_nc*16)
        self.Att3 = Attention_block(F_g=in_nc*4, F_l=in_nc*4, F_int=in_nc*2)
        self.DCR_block33 = D_Block(in_nc*8, in_nc*8)
        self.DCR_block34 = D_Block(in_nc*8, in_nc*8)
        self.up2 = self.make_layer(_up, in_nc*8)
        self.Att2 = Attention_block(F_g=in_nc*2, F_l=in_nc*2, F_int=in_nc)
        self.DCR_block23 = D_Block(in_nc*4, in_nc*4)
        self.DCR_block24 = D_Block(in_nc*4, in_nc*4)
        self.up1 = self.make_layer(_up, in_nc*4)
        self.Att1 = Attention_block(F_g=in_nc, F_l=in_nc, F_int=in_nc//2)
        self.DCR_block13 = D_Block(in_nc*2, in_nc*2)
        self.DCR_block14 = D_Block(in_nc*2, out_nc)
        # self.conv_f = nn.Conv2d(in_channels=in_nc*2, out_channels=out_nc, kernel_size=1, stride=1, padding=0)
        # self.relu2 = nn.PReLU()

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.DCR_block11(x)

        conc1 = self.DCR_block12(out)

        out = self.down1(conc1)

        out = self.DCR_block21(out)

        conc2 = self.DCR_block22(out)

        out = self.down2(conc2)

        out = self.DCR_block31(out)

        conc3 = self.DCR_block32(out)

        conc4 = self.down3(conc3)

        out = self.DCR_block41(conc4)

        out = self.DCR_block42(out)

        out = torch.cat([conc4, out], 1)

        out = self.up3(out)
        out = self.Att3(g=conc3, x=out)

        out = torch.cat([conc3, out], 1)

        out = self.DCR_block33(out)

        out = self.DCR_block34(out)

        out = self.up2(out)
        out = self.Att2(g=conc2, x=out)

        out = torch.cat([conc2, out], 1)

        out = self.DCR_block23(out)

        out = self.DCR_block24(out)

        out = self.up1(out)
        out = self.Att1(g=conc1, x=out)

        out = torch.cat([conc1, out], 1)

        out = self.DCR_block13(out)

        out = self.DCR_block14(out)

        # out = self.relu2(self.conv_f(out))

        return out



import functools
import torch
from torch.nn import init
def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """
    print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
    net.apply(fn)
# 先用UNet试试，如果不行再尝试 att Unet
class Conv2dLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        in_channels += out_channels

        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, states):
        (cell, hidden) = states
        input = torch.cat((hidden, input), dim=1)

        forget_gate = torch.sigmoid(self.forget(input))
        input_gate = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate = torch.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return cell, hidden

class GenerationCore(nn.Module):
    def __init__(self, in_channel):
        super(GenerationCore, self).__init__()
        self.conv_c = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_x = nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.core = Conv2dLSTMCell(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(64, in_channel, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x_q, c_g, h_g):
        # print(x_q.shape, c_g.shape, h_g.shape, u.shape)
        x_q = self.conv_x(x_q)
        c_g = self.conv_c(c_g)
        # h_g = self.conv_r(h_g)
        # print(x_q.shape, c_g.shape, h_g.shape)
        c_g, h_g = self.core(x_q, (c_g, h_g))
        c_g = self.conv(c_g)
        return c_g, h_g


class Net(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, t=10, nb=17, act_mode='BR', nG=5): # nG为混合高斯个数
        super(Net, self).__init__()
        self.channel_trans = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=True)
        # self.channel_trans2 = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=True)
        # self.Unet1 = UNet_att(nc, nc)
        self.Unet1 = UNet_org(nc, nc)
        # self.Unet2 = UNet_org(nc, nc)
        init_weights(self.channel_trans)
        init_weights(self.Unet1)
        # init_weights(self.Unet2)
        self.t = t
        self.generation_core = GenerationCore(in_nc)

        # m_body_ = Recurrent_block(nc)
        # self.image_net_tail = B.conv(nc, out_nc, mode='C', bias=True)

    def forward(self, x, t, train=True): # 非residual
        # b,c,h,w = x.shape
        # noises = torch.zeros(10,b,c,h,w)
        noises = []
        feature = self.Unet1(self.channel_trans(x))
        h_g = torch.zeros_like(feature).cuda()
        c_g = torch.randn_like(x).cuda()
        for i in range(t):
            c_g, h_g = self.generation_core(torch.cat([feature,h_g],1), c_g, h_g) # c 表示 预测的每一步噪声， h时隐藏状态
            noises.append(c_g)
        # c_g = self.Unet2(c_g)
        # feature_c = self.Unet2(self.channel_trans2(c_g))
        # c_g = self.image_net_tail(c_g)
        return x-c_g

# class Net(nn.Module):
#     def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR', nG=5): # nG为混合高斯个数
#         super(Net, self).__init__()
#         self.channel_trans = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=True)
#         self.Unet = UNet_org(nc, nc)
#         init_weights(self.channel_trans)
#         init_weights(self.Unet)
#         self.generation_core = GenerationCore(in_nc)
#
#         # m_body_ = Recurrent_block(nc)
#         self.image_net_tail = B.conv(nc, out_nc, mode='C', bias=True)
#
#     def forward(self, x, t, train=True): # 非residual
#         # b,c,h,w = x.shape
#         # noises = torch.zeros(10,b,c,h,w)
#         noises = []
#         feature = self.Unet(self.channel_trans(x))
#         for i in range(t):
#             if i == 0:
#                 h_g = torch.zeros_like(feature).cuda()
#                 c_g = torch.randn_like(x).cuda()
#             c_g, h_g = self.generation_core(feature, c_g, h_g) # c 表示 预测的每一步噪声， h时隐藏状态
#             noises.append(c_g)
#         return x-c_g, noises
