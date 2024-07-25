import torch.optim
import torch.optim
import torch.optim
import tensorflow as tf
import torch
import torchvision
import torch.optim
import os
import cv2
import numpy as np
from PIL import Image
import s_RDBCAPAnet

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
    data_hazy = data_hazy.resize((576, 576),  resample=Image.LANCZOS)
    feb = feb.resize((576, 576),  resample=Image.LANCZOS)
    gama = gama.resize((576, 576),  resample=Image.LANCZOS)
    zft = zft.resize((576, 576),  resample=Image.LANCZOS)
    white = white.resize((576, 576),  resample=Image.LANCZOS)
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
    dehaze_net = s_RDBCAPAnet.dehaze_net().cuda()
    #加载用于比较的两个模型
    dehaze_net.load_state_dict(torch.load('snapshots/dehazerRDBCAPAbestssimnet.pth'))
    clean_image = dehaze_net(feb, white, gama)
    #存储结果的路径
    torchvision.utils.save_image(clean_image,
                                 "result/" + image_path.split("/")[-1])


    return clean_image
def dehaze_image1(image_path, feb_path, gama_path, zft_path,white_path):

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
    data_hazy = data_hazy.resize((576, 576),  resample=Image.LANCZOS)
    feb = feb.resize((576, 576),  resample=Image.LANCZOS)
    gama = gama.resize((576, 576),  resample=Image.LANCZOS)
    zft = zft.resize((576, 576),  resample=Image.LANCZOS)
    white = white.resize((576, 576),  resample=Image.LANCZOS)
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
    dehaze_net = s_RDBCAPAnet.dehaze_net().cuda()
    #用于加载比较后性能最优的模型
    dehaze_net.load_state_dict(torch.load('snapshots/dehazerRDBCAPAbestssimnet.pth'))
    clean_image = dehaze_net(feb, white, gama)
    #用于存储性能最优模型的结果
    torchvision.utils.save_image(clean_image,
                                 "result/" + image_path.split("/")[-1])


    return clean_image

if __name__ == '__main__':

    #请加载测试集相关
    #可以看备忘录所需数据集，与之一一对应
    #原数据集路径
    img_dir = os.path.join('adjust/nyuhaze500/hazy/')
    #rdb处理后数据路径
    feb_dir = os.path.join('adjust/nyuhaze500/feb/')
    #gama处理后数据路径
    gama_dir = os.path.join('adjust/nyuhaze500/gama/')
    #直方图处理后数据路径
    zft_dir = os.path.join('adjust/nyuhaze500/zft/')
    #白平衡出理后数据路径
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
    #存放处理前源数据，即将结果图形状改为与原来一致
    orig_dir = os.path.join('D:/ws/KTMDA-DehazeNet-main/KTMDA/haze-tr/')
    # 存放需要处理的图片
    img_dir = os.path.join('D:/ws/KTMDA-DehazeNet-main/KTMDA/io-train/')
    img_list = os.listdir(orig_dir)
    for img in img_list:
        # print(img)
        img1 = cv2.imread(orig_dir + img)
        a = img1.shape[0]
        b = img1.shape[1]
        img2 = cv2.imread(img_dir + img)
        img2 = cv2.resize(img2, (b, a))
        #图像处理后存放地址
        cv2.imwrite("D:/ws/KTMDA-DehazeNet-main/KTMDA/do/" + img, img2)

    #此段计算处理后图像与清晰图像的psnr值
    for i in range(1, 51):
        img1 = cv2.imread("D:/ws/KTMDA-DehazeNet-main/KTMDA/outdoor/test/clear/GT_" + str(i) + ".png")
        # 输入清晰图像
        img2 = cv2.imread("D:/ws/KTMDA-DehazeNet-main/KTMDA/do_psnr/GT_" + str(i) + ".png")
        # 输入输出图像
        psnr_re = tf.image.psnr(img1, img2, 255)
        print('%.3f' % psnr_re)
    print('==' * 30)
    for i in range(1, 51):
        img1 = cv2.imread("D:/ws/KTMDA-DehazeNet-main/KTMDA/outdoor/test/clear/GT_" + str(i) + ".png")
        img2 = cv2.imread("D:/ws/KTMDA-DehazeNet-main/KTMDA/do_psnr/GT_" + str(i) + ".png")
        ssim_re = tf.image.ssim(img1, img2, 255)
        print('%.3f' % ssim_re)
    #通过比较两模型的psnr和ssim值选出最优模型

    #以下步骤为用最优模型给你想要的数据集去雾，即训练集相关
    #以下步骤最好在比较两个模型性能时省略，自己加#
    #找到最优模型时，只用下面部分，上面可以省略
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
        dehaze_image1(image_path, feb_path, gama_path, zft_path,white_path)
        print(image, "done!")
    #存放原始图像路径
    orig_dir = os.path.join('D:/ws/KTMDA-DehazeNet-main/KTMDA/haze-tr/')
    # 存放需要处理的图片
    img_dir = os.path.join('D:/ws/KTMDA-DehazeNet-main/KTMDA/io-train/')
    img_list = os.listdir(orig_dir)
    for img in img_list:
        # print(img)
        img1 = cv2.imread(orig_dir + img)
        a = img1.shape[0]
        b = img1.shape[1]
        img2 = cv2.imread(img_dir + img)
        img2 = cv2.resize(img2, (b, a))
        #图像处理后存放地址，该结果即最终结果
        cv2.imwrite("D:/ws/KTMDA-DehazeNet-main/KTMDA/do/" + img, img2)