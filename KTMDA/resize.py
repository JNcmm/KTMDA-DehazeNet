import os
import cv2

#用来调整图片尺寸大小

orig_dir = os.path.join('E:/yanjiushengstudy/PyCharm2020.1/chenchen/adjust/realimage/hazy/')
img_dir = os.path.join('E:/yanjiushengstudy/PyCharm2020.1/chenchen/adjust/realimage/gama1/')
img_list = os.listdir(orig_dir)

for img in img_list:
    # print(img)
    img1 = cv2.imread(orig_dir + img)
    a = img1.shape[0]
    b = img1.shape[1]
    img2 = cv2.imread(img_dir + img)
    img2 = cv2.resize(img2,(b,a))
    cv2.imwrite("E:/yanjiushengstudy/PyCharm2020.1/chenchen/adjust/realimage/gama1/"+img,img2)
    # cv2.imwrite("E:\\yanjiushengstudy\PyCharm 2020.1\\PyTorch-Image-Dehazing-master\\对比实验结果\\zft用python\\gama+原图+zft+dcp+unet跳跃连接\\")
