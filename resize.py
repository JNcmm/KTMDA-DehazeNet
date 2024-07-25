import os
import cv2

#用来调整图片尺寸大小
#存放原始图片
orig_dir = os.path.join('D:/ws/KTMDA-DehazeNet-main/KTMDA/haze-tr/')
#存放处理后图片
img_dir = os.path.join('D:/ws/KTMDA-DehazeNet-main/KTMDA/io-train/')
img_list = os.listdir(orig_dir)

for img in img_list:
    # print(img)
    img1 = cv2.imread(orig_dir + img)
    a = img1.shape[0]
    b = img1.shape[1]
    img2 = cv2.imread(img_dir + img)
    img2 = cv2.resize(img2,(b,a))
    cv2.imwrite("D:/ws/KTMDA-DehazeNet-main/KTMDA/do/"+img,img2)

