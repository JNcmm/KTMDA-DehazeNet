import torch
import torch.nn as nn
import math
import numpy as np
import cv2
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import time




class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, res_blocks=6):
        super(MakeDense, self).__init__()
        self.dehaze = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(18))
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        out = self.dehaze(out)
        return out

class MakeDense1(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, res_blocks=6):
        super(MakeDense1, self).__init__()
        self.dehaze1 = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze1.add_module('res%d' % i, ResidualBlock(27))
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        out = self.dehaze1(out)
        return out

class MakeDense2(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, res_blocks=6):
        super(MakeDense2, self).__init__()
        self.dehaze2 = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze2.add_module('res%d' % i, ResidualBlock(36))
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        out = self.dehaze2(out)
        return out

class MakeDense3(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, res_blocks=6):
        super(MakeDense3, self).__init__()
        self.dehaze3 = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze3.add_module('res%d' % i, ResidualBlock(15))
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        # out = self.dehaze3(out)
        return out

class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()

        _in_channels = in_channels
        modules = []
        # for i in range(num_dense_layer):
        #     modules.append(MakeDense(_in_channels, growth_rate))
        #     _in_channels += growth_rate
        modules.append(MakeDense(_in_channels, growth_rate))
        _in_channels += growth_rate

        modules.append(MakeDense1(_in_channels, growth_rate))
        _in_channels += growth_rate

        modules.append(MakeDense2(_in_channels, growth_rate))
        _in_channels += growth_rate

        modules.append(MakeDense3(_in_channels, growth_rate))
        _in_channels += growth_rate

        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class MakeDense4(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense4, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class RDB1(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB1, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense4(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class PALayer2(nn.Module):
    def __init__(self, channel):
        super(PALayer2, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 2, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, 1, 1, padding=0, bias=True),
                nn.ReLU(inplace=True)
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.ReLU(inplace=True)
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


# class Down(nn.Module):
#     def __init__(self, in_channels, kernel_size=3, stride=2):
#         super(Down, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
#         # self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         # out = F.relu(self.conv2(out))
#         return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self):
        super().__init__()
        self.maxpool_conv = nn.Sequential(  # ????
            nn.MaxPool2d(2)
            # DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class dehaze_net(nn.Module):
    def __init__(self, depth_rate=16, kernel_size=3, num_dense_layer=4, growth_rate=16, res_blocks=6):
        super(dehaze_net, self).__init__()
        self.rdb_in = RDB(6, num_dense_layer, 6)
        self.rdb_in1 = RDB1(12, num_dense_layer, 12)
        self.ca1 = CALayer(16)
        self.palayer1 = PALayer(16)
        self.down1 = Down()
        self.up1 = UpsampleConvLayer(6, 6, 3, 2)
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(12, 12, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(12, 12, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(24, 12, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(24, 12, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(48, 12, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(12, 6, 1, 1, 0, bias=True)

        self.e_conv = nn.Conv2d(6, 16, 3, 1, 1, bias=True)
        self.e_conv10 = nn.Conv2d(16, 64, 3, 1, 1, bias=True)
        self.e_conv9 = nn.Conv2d(6, 3, 3, 1, 1, bias=True)
        self.e_conv8 = nn.Conv2d(6, 6, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(64, 16, 3, 1, 1, bias=True)
        self.e_conv11 = nn.Conv2d(16, 6, 3, 1, 1, bias=True)

        self.dilate1 = nn.Conv2d(64, 16, 3, 1, 1,dilation=1 ,bias=True)
        self.dilate2 = nn.Conv2d(64, 16, 3, 1, 2,dilation=2 ,bias=True)
        self.dilate3 = nn.Conv2d(64, 16, 3, 1, 4,dilation=4 ,bias=True)
        self.dilate4 = nn.Conv2d(64, 16, 3, 1, 8,dilation=8 ,bias=True)

        self.dehaze2 = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze2.add_module('res%d' % i, ResidualBlock(24))
        self.dehaze3 = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze3.add_module('res%d' % i, ResidualBlock(48))

    def forward(self, x, y, z):

        x = x.cpu().detach().numpy()
        z = z.cpu().detach().numpy()
        l = np.concatenate((x, z), axis=1)
        l = torch.from_numpy(l).float()
        l = l.cuda()
        rdb = self.rdb_in(l)
        u = self.relu(self.e_conv(rdb))
        ca = self.ca1(u)
        pa = self.palayer1(ca)
        rdb = u + pa
        # rdb = self.rdb_in(rdb)
        rdb = self.relu(self.e_conv10(rdb))
        di1 = self.relu(self.dilate1(rdb))
        di2 = self.relu(self.dilate2(rdb))
        di3 = self.relu(self.dilate3(rdb))
        di4 = self.relu(self.dilate4(rdb))
        di = torch.cat((di1,di2,di3,di4), 1)
        di = self.relu(self.e_conv7(di))
        di = self.relu(self.e_conv11(di))
        l1 = l + di
        t1 = self.down1(l1)

        # u1 = self.relu(self.e_conv(t1))
        rdb1 = self.rdb_in(t1)
        u1 = self.relu(self.e_conv(rdb1))
        ca1 = self.ca1(u1)
        pa1 = self.palayer1(ca1)
        rdb1 = u1 + pa1
        # rdb1 = self.rdb_in(rdb1)
        rdb1 = self.relu(self.e_conv10(rdb1))
        di11 = self.relu(self.dilate1(rdb1))
        di22 = self.relu(self.dilate2(rdb1))
        di33 = self.relu(self.dilate3(rdb1))
        di44 = self.relu(self.dilate4(rdb1))
        di0 = torch.cat((di11,di22,di33,di44), 1)
        di0 = self.relu(self.e_conv7(di0))
        di0 = self.relu(self.e_conv11(di0))
        l2 = t1 + di0
        t2 = self.down1(l2)

        # u2 = self.relu(self.e_conv(t2))
        rdb2 = self.rdb_in(t2)
        u2 = self.relu(self.e_conv(rdb2))
        ca2 = self.ca1(u2)
        pa2 = self.palayer1(ca2)
        rdb2 = u2 + pa2
        # rdb2 = self.rdb_in(rdb2)
        rdb2 = self.relu(self.e_conv10(rdb2))
        di111 = self.relu(self.dilate1(rdb2))
        di222 = self.relu(self.dilate2(rdb2))
        di333 = self.relu(self.dilate3(rdb2))
        di444 = self.relu(self.dilate4(rdb2))
        di00 = torch.cat((di111, di222, di333,di444), 1)
        di00 = self.relu(self.e_conv7(di00))
        di00 = self.relu(self.e_conv11(di00))
        l3 = t2 + di00

        k1 = self.up1(l3)
        k1 = F.upsample(k1, l2.size()[2:], mode='bilinear')
        k2 = self.relu(self.e_conv8(k1))
        lres1 = l2 - k2
        k1 = torch.cat((k1,l2), 1)
        b1 = self.relu(self.e_conv1(k1))
        b2 = self.relu(self.e_conv2(b1))
        concatb1 = torch.cat((b1, b2), 1)
        concatb1 = self.dehaze2(concatb1)
        b3 = self.relu(self.e_conv3(concatb1))
        concatb2 = torch.cat((b2, b3), 1)
        concatb2 = self.dehaze2(concatb2)
        b4 = self.relu(self.e_conv4(concatb2))
        concatb3 = torch.cat((b1, b2, b3, b4), 1)
        concatb3 = self.dehaze3(concatb3)
        k1 = self.relu(self.e_conv5(concatb3))
        k1 = self.relu(self.e_conv6(k1))
        k1 = torch.cat((k1, lres1), 1)
        k1 = self.rdb_in1(k1)
        k1 = self.relu(self.e_conv6(k1))
        # b5 = self.relu(self.e_conv1(k1))
        # b6 = self.relu(self.e_conv2(b5))
        # concatb11 = torch.cat((b5, b6), 1)
        # concatb11 = self.dehaze2(concatb11)
        # b7 = self.relu(self.e_conv3(concatb11))
        # concatb22 = torch.cat((b6, b7), 1)
        # concatb22 = self.dehaze2(concatb22)
        # b8 = self.relu(self.e_conv4(concatb22))
        # concatb33 = torch.cat((b5, b6, b7, b8), 1)
        # concatb33 = self.dehaze3(concatb33)
        # k1 = self.relu(self.e_conv5(concatb33))
        # k1 = self.relu(self.e_conv6(k1))

        k3 = self.up1(k1)
        k3 = F.upsample(k3, l1.size()[2:], mode='bilinear')
        k3 = self.relu(self.e_conv8(k3))
        lres2 = l1 - k3
        k3 = torch.cat((k3,l1), 1)
        a1 = self.relu(self.e_conv1(k3))
        a2 = self.relu(self.e_conv2(a1))
        concatt1 = torch.cat((a1, a2), 1)
        concatt1 = self.dehaze2(concatt1)
        a3 = self.relu(self.e_conv3(concatt1))
        concatt2 = torch.cat((a2, a3), 1)
        concatt2 = self.dehaze2(concatt2)
        a4 = self.relu(self.e_conv4(concatt2))
        concatt3 = torch.cat((a1, a2, a3, a4), 1)
        concatt3 = self.dehaze3(concatt3)
        k3 = self.relu(self.e_conv5(concatt3))
        k3 = self.relu(self.e_conv6(k3))
        k3 = torch.cat((k3, lres2), 1)
        k3 = self.rdb_in1(k3)
        k3 = self.relu(self.e_conv6(k3))
        k3 = self.relu(self.e_conv9(k3))

        # a5 = self.relu(self.e_conv1(k3))
        # a6 = self.relu(self.e_conv2(a1))
        # concatt11 = torch.cat((a5, a6), 1)
        # concatt11 = self.dehaze2(concatt11)
        # a7 = self.relu(self.e_conv3(concatt11))
        # concatt22 = torch.cat((a6, a7), 1)
        # concatt22 = self.dehaze2(concatt22)
        # a8 = self.relu(self.e_conv4(concatt22))
        # concatt33 = torch.cat((a5, a6, a7, a8), 1)
        # concatt33 = self.dehaze3(concatt33)
        # k3 = self.relu(self.e_conv5(concatt33))
        # k3 = self.relu(self.e_conv6(k3))

        # clean_image = self.relu(self.e_conv9(k3))

        return k3

