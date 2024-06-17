import torch
import torchvision
import torch.optim
import os
import cv2;
import math;
import numpy as np;
from PIL import Image
import RDBCAPAnet


def dehaze_image(image_path, feb_path, gama_path, zft_path,white_path):

    data_hazy = Image.open(image_path)
    feb = Image.open(feb_path)
    gama = Image.open(gama_path)
    zft = Image.open(zft_path)
    white = Image.open(white_path)
    data_hazy = data_hazy.convert("RGB")
    feb = feb.convert("RGB")
    gama = gama.convert("RGB")
    zft = zft.convert("RGB")
    white = white.convert("RGB")
    data_hazy = data_hazy.resize((576, 576), Image.ANTIALIAS)
    feb = feb.resize((576, 576), Image.ANTIALIAS)
    gama = gama.resize((576, 576), Image.ANTIALIAS)
    zft = zft.resize((576, 576), Image.ANTIALIAS)
    white = white.resize((576, 576), Image.ANTIALIAS)
    data_hazy = (np.asarray(data_hazy) / 255.0)
    feb = (np.asarray(feb) / 255.0)
    gama = (np.asarray(gama) / 255.0)
    zft = (np.asarray(zft) / 255.0)
    white = (np.asarray(white) / 255.0)
    data_hazy = torch.from_numpy(data_hazy).float()
    feb = torch.from_numpy(feb).float()
    gama = torch.from_numpy(gama).float()
    zft = torch.from_numpy(zft).float()
    white = torch.from_numpy(white).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    feb = feb.permute(2, 0, 1)
    gama = gama.permute(2, 0, 1)
    zft = zft.permute(2, 0, 1)
    white = white.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)
    feb = feb.cuda().unsqueeze(0)
    gama = gama.cuda().unsqueeze(0)
    zft = zft.cuda().unsqueeze(0)
    white = white.cuda().unsqueeze(0)
    dehaze_net = RDBCAPAnet.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load('snapshots/dehazerRDBCAPAbestssimnet.pth'))
    clean_image = dehaze_net(feb, white, gama)
    torchvision.utils.save_image(clean_image,
                                 "result/" + image_path.split("/")[-1])


    return clean_image


if __name__ == '__main__':


    img_dir = os.path.join('adjust/nyuhaze500/hazy/')
    feb_dir = os.path.join('adjust/nyuhaze500/feb/')
    gama_dir = os.path.join('adjust/nyuhaze500/gama/')
    zft_dir = os.path.join('adjust/nyuhaze500/zft/')
    white_dir = os.path.join('adjust/nyuhaze500/white/')
    test_list = os.listdir(img_dir)
    for image in test_list:
        feb_path = feb_dir + image
        image_path = img_dir + image
        gama_path = gama_dir + image
        zft_path = zft_dir + image
        white_path = white_dir + image
        dehaze_image(image_path, feb_path, gama_path, zft_path,white_path)
        print(image, "done!")
