import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import newdataloader
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import Perceptual
import RDBCAPAnet
import gamardbnet
import dataloadergama
import torch.nn.functional as F
import CR
from math import log10
from skimage.metrics import structural_similarity as compare_ssim


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [compare_ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list


def validation(net, val_data_loader, device):
    psnr_list = []
    ssim_list = []

    for batch_id, (img_orig, img_haze, rdb, gama,zft) in enumerate(val_data_loader):

        with torch.no_grad():
            img_orig = img_orig.to(device)

            img_haze = img_haze.to(device)

            rdb = rdb.to(device)

            gama = gama.to(device)

            zft = zft.to(device)

            clean_image = net(img_haze, zft)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(clean_image, img_orig))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(clean_image, img_orig))

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim

def train(config):

    dehaze_net = gamardbnet.dehaze_net().cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = dataloadergama.dehazing_loader(config.orig_images_path,
                                               config.hazy_images_path,
                                                  config.rdb_path,
                                                  config.gama_path,
                                                  config.zft_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True,
                              num_workers=config.num_workers)

    # old_val_psnr, old_val_ssim = validation(dehaze_net, train_loader, device)
    old_val_psnr, old_val_ssim = validation(dehaze_net, train_loader, device)
    print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

    criterion = nn.MSELoss().cuda()
    perception = Perceptual.PerceptualLoss().cuda()
    contrastloss = CR.ContrastLoss().cuda()
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):

        print('==' * 30)
        print("epoch:" + str(epoch))

        for iteration, (img_orig, img_haze, rdb, gama, zft) in enumerate(train_loader):

            img_orig = img_orig.to(device)

            img_haze = img_haze.to(device)

            rdb = rdb.to(device)

            gama = gama.to(device)

            zft = zft.to(device)

            dehaze_net.train()

            clean_image = dehaze_net(img_haze, zft)

            loss1 = F.smooth_l1_loss(clean_image, img_orig)
            loss2 = perception(clean_image, img_orig)
            # loss4 = -ssim_loss(clean_image, img_orig)
            loss3 = contrastloss(clean_image, img_orig, img_haze)

            loss = loss1 + 0.1 * loss2 + 0.1 * loss3

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())

            # --- Save the network parameters --- #
        torch.save(dehaze_net.state_dict(), config.snapshots_folder + ("dehazer2RDBCAPA" + str(epoch) + "net.pth"))

        # # --- Use the evaluation model in testing --- #
        # dehaze_net.eval()

        val_psnr, val_ssim = validation(dehaze_net, train_loader, device)
        print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
        # --- update the network weight --- #
        if val_psnr >= old_val_psnr:
            torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazerRDBCAPAbestpsnrnet.pth")
            print("epoch==" + str(epoch) + "bestpsnr")
            old_val_psnr = val_psnr
        if val_ssim >= old_val_ssim:
            torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazerRDBCAPAbestssimnet.pth")
            print("epoch==" + str(epoch) + "bestssim")
            old_val_ssim = val_ssim



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    a=1
    b=1

    parser.add_argument('--orig_images_path', type=str, default="SOTS/outdoor/clear/")
    parser.add_argument('--hazy_images_path', type=str, default="SOTS/outdoor/hazy/")
    parser.add_argument('--gama_path', type=str, default="SOTS/outdoor/clear/")
    parser.add_argument('--rdb_path', type=str, default="SOTS/outdoor/hazy/")
    parser.add_argument('--zft_path', type=str, default="SOTS/outdoor/zft/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")


    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)


    train(config)
