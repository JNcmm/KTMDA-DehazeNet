import cv2
import numpy as np
import os


img_dir = os.path.join("D:\\ws\\KTMDA-DehazeNet-main\\KTMDA\\outdoor\\train\\haze")
img_list = os.listdir(img_dir)

for a in img_list:
    # 读取图像
    # print(a)
    img = cv2.imread("D:\\ws\\KTMDA-DehazeNet-main\\KTMDA\\outdoor\\train\\haze\\"+ a)
    # print(img)
    # 计算图像中所有像素的平均值
    avg_bgr = cv2.mean(img)
    # 计算平均值的灰度值
    gray_value = sum(avg_bgr[:3]) / 3
    # 计算调整系数
    coefficients = [gray_value / v for v in avg_bgr[:3]]
    # 调整每个像素的颜色值
    img_white_balance = np.zeros_like(img)
    for i in range(3):
        img_white_balance[:, :, i] = np.uint8(np.clip(coefficients[i] * img[:, :, i], 0, 255))
    # 保存结果
    cv2.imwrite("D:\\ws\\KTMDA-DehazeNet-main\\KTMDA\\withe-result\\" +a, img_white_balance)