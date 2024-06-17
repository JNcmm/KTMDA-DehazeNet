import torch
import torchvision
import torch.optim
import os
import numpy as np;
from PIL import Image
import rdbresblock

def dehaze_image(image_path):

    data_hazy = Image.open(image_path)
    data_hazy = data_hazy.convert("RGB")
    data_hazy = data_hazy.resize((576, 576), Image.ANTIALIAS)
    data_hazy = (np.asarray(data_hazy) / 255.0)
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)
    dehaze_net = rdbresblock.dehaze_net().cuda()

    dehaze_net.load_state_dict(torch.load('snapshots/dehazerrdbnet.pth'))

    clean_image = dehaze_net(data_hazy)
    torchvision.utils.save_image(clean_image,
                                 "result/" + image_path.split("/")[-1])


    return clean_image


if __name__ == '__main__':


    img_dir = os.path.join('adjust/testdata/hazy/')
    test_list = os.listdir(img_dir)
    for image in test_list:
        image_path = img_dir + image
        dehaze_image(image_path)
        print(image, "done!")
