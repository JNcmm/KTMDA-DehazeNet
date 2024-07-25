import tensorflow as tf
import cv2
#用来求PSNR和SSIM的
for i in range(1, 51):#E:\yanjiushengstudy\PyCharm 2020.1\PyTorch-Image-Dehazing-master\result\SPAnet\result
    img1 = cv2.imread("D:/ws/KTMDA-DehazeNet-main/KTMDA/outdoor/test/clear/GT_" + str(i) + ".png")
    #输入清晰图像
    img2 = cv2.imread("D:/ws/KTMDA-DehazeNet-main/KTMDA/do_psnr/GT_" + str(i) + ".png")
    #输入输出图像
    psnr_re = tf.image.psnr(img1, img2, 255)
    print('%.3f' % psnr_re)

print('==' * 30)

for i in range(1, 51):
    img1 = cv2.imread("D:/ws/KTMDA-DehazeNet-main/KTMDA/outdoor/test/clear/GT_" + str(i) + ".png")
    img2 = cv2.imread("D:/ws/KTMDA-DehazeNet-main/KTMDA/do_psnr/GT_" + str(i) + ".png")
    ssim_re = tf.image.ssim(img1, img2, 255)
    print('%.3f' % ssim_re)






