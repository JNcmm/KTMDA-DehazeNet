import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2



def populate_train_list(orig_images_path, hazy_images_path, rdb_path, gama_path,zft_path,dcp_path):
# def populate_train_list(orig_images_path, hazy_images_path):

    train_list = []
    val_list = []

    image_list_haze = glob.glob(hazy_images_path + "*.jpg")

    tmp_dict = {}

    for image in image_list_haze:
        image = image.split("\\")[-1]
        # image = image.split("\\")[-1]
        key = image.split("_")[0] + "_" + image.split("_")[1]
        # key = image.split("_")[0] + "_" + image.split("_")[1] + ".jpg"
        if key in tmp_dict.keys():
            tmp_dict[key].append(image)
        else:
            tmp_dict[key] = []
            tmp_dict[key].append(image)

    train_keys = []
    val_keys = []

    len_keys = len(tmp_dict.keys())
    for i in range(len_keys):
        # if i < len_keys * 9 / 10:
        if i < len_keys:
            train_keys.append(list(tmp_dict.keys())[i])
        else:
            val_keys.append(list(tmp_dict.keys())[i])

    for key in list(tmp_dict.keys()):

        if key in train_keys:
            for hazy_image in tmp_dict[key]:
                train_list.append([orig_images_path + key, hazy_images_path + hazy_image, rdb_path + key, gama_path + key, zft_path + key,dcp_path + key])
                # train_list.append([orig_images_path + key, hazy_images_path + hazy_image])



        else:
            for hazy_image in tmp_dict[key]:
                val_list.append([orig_images_path + key, hazy_images_path + hazy_image, rdb_path + key, gama_path + key, dcp_path + key])
                # val_list.append([orig_images_path + key, hazy_images_path + hazy_image])


    random.shuffle(train_list)
    # random.shuffle(val_list)

    return train_list, val_list


class dehazing_loader(data.Dataset):

    def __init__(self, orig_images_path, hazy_images_path, rdb_path,gama_path,zft_path,dcp_path, mode='train'):
    # def __init__(self, orig_images_path, hazy_images_path, mode='train'):

        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path, rdb_path, gama_path,zft_path,dcp_path)
        # self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path)


        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_orig_path, data_hazy_path, rdb_path, gama_path, zft_path, dcp_path = self.data_list[index]
        # data_orig_path, data_hazy_path = self.data_list[index]


        data_orig = Image.open(data_orig_path)
        # cv2.cvtColor(data_orig,cv2.COLOR_BGR2GRAY)
        data_orig = data_orig.convert("RGB")
        data_hazy = Image.open(data_hazy_path)
        data_hazy = data_hazy.convert("RGB")
        gama = Image.open(gama_path)
        gama = gama.convert("RGB")
        dcp = Image.open(rdb_path)
        dcp = dcp.convert("RGB")
        zft = Image.open(zft_path)
        zft = zft.convert("RGB")
        a = Image.open(dcp_path)
        a = a.convert("RGB")
        data_orig = data_orig.resize((576, 576), resample=Image.LANCZOS)
        data_hazy = data_hazy.resize((576, 576), resample=Image.LANCZOS)
        gama = gama.resize((576, 576), resample=Image.LANCZOS)
        dcp = dcp.resize((576, 576),resample=Image.LANCZOS)
        zft = zft.resize((576, 576), resample=Image.LANCZOS)
        a = a.resize((576, 576), resample=Image.LANCZOS)

        data_orig = (np.asarray(data_orig) / 255.0)
        data_hazy = (np.asarray(data_hazy) / 255.0)
        zft = (np.asarray(zft) / 255.0)
        dcp = (np.asarray(dcp) / 255.0)
        gama = (np.asarray(gama) / 255.0)
        a = (np.asarray(a) / 255.0)

        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()
        gama = torch.from_numpy(gama).float()
        rdb = torch.from_numpy(dcp).float()
        zft = torch.from_numpy(zft).float()
        a = torch.from_numpy(a).float()

        # data_orig= torch.reshape(data_orig,shape=[576,576,1])
        # data_hazy= torch.reshape(data_hazy,shape=[576,576,1])

        # return data_orig.permute(1,0), data_hazy.permute(1,0)
        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1), rdb.permute(2,0,1), gama.permute(2, 0, 1),  zft.permute(2, 0, 1), a.permute(2, 0, 1)
        # return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)


    # return data_orig, data_hazy

    def __len__(self):
        return len(self.data_list)

