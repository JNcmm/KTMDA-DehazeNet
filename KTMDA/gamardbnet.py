import torch
import torch.nn as nn
import math
import numpy as np
import cv2
import torch.nn.functional as F

class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, res_blocks=6):
        super(MakeDense, self).__init__()
        self.dehaze = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(32))
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        out = self.dehaze(out)
        return out

class MakeDense1(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=5, res_blocks=6):
        super(MakeDense1, self).__init__()
        self.dehaze1 = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze1.add_module('res%d' % i, ResidualBlock(48))
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        out = self.dehaze1(out)
        return out

class MakeDense2(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=7, res_blocks=6):
        super(MakeDense2, self).__init__()
        self.dehaze2 = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze2.add_module('res%d' % i, ResidualBlock(64))
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
            self.dehaze3.add_module('res%d' % i, ResidualBlock(80))
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        out = self.dehaze3(out)
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


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
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
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class dehaze_net(nn.Module):
    def __init__(self,res_blocks=6):
        super(dehaze_net, self).__init__()

        self.ca = CALayer(15)
        self.palayer = PALayer(15)
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(15, 15, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(15, 15, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(30, 15, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(30, 15, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(60, 15, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(15, 3, 1, 1, 0, bias=True)
        self.dehaze2 = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze2.add_module('res%d' % i, ResidualBlock(30))
        self.dehaze3 = nn.Sequential()
        for i in range(1, res_blocks):
            self.dehaze3.add_module('res%d' % i, ResidualBlock(60))

    def forward(self, x, y):

        source = []
        source.append(x)
        # gama变换
        img_haze = x.cpu().detach().numpy()
        img2 = np.power(img_haze, 2)
        img2 = torch.from_numpy(img2).float()
        img3 = np.power(img_haze, 3)
        img3 = torch.from_numpy(img3).float()
        img4 = np.power(img_haze, 4)
        img4 = torch.from_numpy(img4).float()
        img5 = np.power(img_haze, 5)
        img5 = torch.from_numpy(img5).float()

        img2 = img2.cpu().detach().numpy()
        img3 = img3.cpu().detach().numpy()
        img4 = img4.cpu().detach().numpy()
        img5 = img5.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        x = x.cpu().detach().numpy()

        l = np.concatenate((img2, img3, img4, img5, y), axis=1)
        l = torch.from_numpy(l).float()
        x = torch.from_numpy(x).float()
        l = l.cuda()
        x = x.cuda()

        b1 = self.relu(self.e_conv1(l))
        b2 = self.relu(self.e_conv2(b1))
        concatb1 = torch.cat((b1, b2), 1)
        concatb1 = self.dehaze2(concatb1)
        b3 = self.relu(self.e_conv3(concatb1))
        concatb2 = torch.cat((b2, b3), 1)
        concatb2 = self.dehaze2(concatb2)
        b4 = self.relu(self.e_conv4(concatb2))
        concatb3 = torch.cat((b1, b2, b3, b4), 1)
        concatb3 = self.dehaze3(concatb3)
        k = self.e_conv5(concatb3)
        w = self.ca(k)
        out = self.palayer(w)
        out = self.e_conv6(out)
        # out = out + x
        return out

