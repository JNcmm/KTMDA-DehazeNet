import tensorflow as tf
import cv2
#用来求PSNR和SSIM的
for i in range(1, 51):#E:\yanjiushengstudy\PyCharm 2020.1\PyTorch-Image-Dehazing-master\result\SPAnet\result
    img1 = cv2.imread("E:/yanjiushengstudy/PyCharm2020.1/chenchen/adjust/testdata/clear/GT_" + str(i) + ".jpg")
    img2 = cv2.imread("E:/yanjiushengstudy/PyCharm2020.1/chenchen/adjust/testdata/gama/GT_" + str(i) + ".jpg")
    psnr_re = tf.image.psnr(img1, img2, 255)
    print('%.3f' % psnr_re)

print('==' * 30)

for i in range(1, 51):
    img1 = cv2.imread("E:/yanjiushengstudy/PyCharm2020.1/chenchen/adjust/testdata/clear/GT_" + str(i) + ".jpg")
    img2 = cv2.imread("E:/yanjiushengstudy/PyCharm2020.1/chenchen/adjust/testdata/gama/GT_" + str(i) + ".jpg")
    ssim_re = tf.image.ssim(img1, img2, 255)
    print('%.3f' % ssim_re)






