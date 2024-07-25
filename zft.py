import numpy as np
import cv2
import os


# 全局直方图均衡化
def hisEqulColor1(img):
    image_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    img = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return img

if __name__ == '__main__':

    img_dir = os.path.join('D:/ws/KTMDA-DehazeNet-main/KTMDA/outdoor/train/haze/')
    img_list = os.listdir(img_dir)
    for img in img_list:
        # print(img)
        img1 = cv2.imread(img_dir + img)
        # # img1 = img.copy()
        res1 = hisEqulColor1(img1)
        cv2.imwrite("D:\\ws\\KTMDA-DehazeNet-main\\KTMDA\\zft_train\\" + img, res1)
