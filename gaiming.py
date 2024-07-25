import re
import os
import shutil

img_dir = os.path.join('D:/ws/KTMDA-DehazeNet-main/KTMDA/outdoor1/')
img_list = os.listdir(img_dir)
print(img_list)
a=1
i=1
for img in img_list:
    # img_name = img.split('.')[0]
    # img_name = img.split('GT')
    # img_name = img.split('GT_')
    # img_name = img.split('_')
    # img_name = img
    #
    # print(img_name)
    print(img)
    new_img_name = 'GT_' + str(i)
    i=i+1
    print(new_img_name)
    # new_img_name = 'GT_' + img_name
    new_img = new_img_name.split('.')[0] + '.jpg'
    print(new_img)
    # # print(img)
    # os.rename(img_dir + str(a)+"_ahq"+".jpg", img_dir + new_img)
    os.rename(img_dir + img, img_dir + new_img)
    # os.rename(img_dir +"GT_"+str(a)+".jpg", img_dir + new_img)
    a=a+1

# print(img_list)