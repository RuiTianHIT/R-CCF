import numpy as np
import cv2
import os
import PIL.Image as Image

file = open("D:\couple_with_mix_box\MISTBOX/test.txt")
image_id = []
for line in file.readlines():
    line = line.strip("\n")
    image_id.append(line)



a = 0
image_id_tatal = []
image_pred_box_class = []
# 目前这里面的图像索引的key已经搭建完成  0 1 4 15 16 32...
dict_info_pred = {}
sub_image_pred = {}
deal_dict = {}

with open("D:/couple_with_mix_box/MISTBOX/box.txt", "r") as f:  # 打开文件
    x = 0
    for line in f.readlines():
        if x % 9 == 0:
            x = 0
            image_id_tatal.clear()
            image_id_tatal.append("_")
        line = line.strip('\n')
        line = line.replace("(", "")
        line = line.replace(")", "")
        line = line.replace(" ", "")
        if a % 9 == 0:
            for j in line.split(","):
                dict_info_pred.update({j: ""})
                image_id_tatal.append(j)
            x = x + 1
        else:
            b = 0
            c = 0
            image_pred_box_class.append(line)
            line = line.split(",")
            del line[-1]
            for i in line:
                if i.isalpha():
                    b = b + 1
                if i.isdigit():
                    c = c + 1
            for o in range(b):
                # 对一行的信息打包成字典
                pred_class = line[-b+o]
                pred_class = pred_class+str(o)
                pred_box_info = line[4*o:4*(o+1)]
                sub_image_pred.update({pred_class: pred_box_info})
            deal_dict = sub_image_pred.copy()
            dict_info_pred.update({image_id_tatal[x]: deal_dict})
            sub_image_pred.clear()
            x = x + 1
        a = a + 1
    a = 0
# # 搭建的训练集trainloader
# # 训练集是从弱监督检测到的物体中进行crop，crop的尺寸是32*32，在检测到的box内进行裁剪
# # 这个循环有点多余了，只是从list转换成了数组，没事，先放着吧到时候可以删除***********
new_list = []
center_ = []
for info in dict_info_pred:
    for sub_info in dict_info_pred[info]:
        for str_box_info in dict_info_pred[info][sub_info]:
            new_list.append(int(str_box_info))
        center_ = new_list.copy()
        dict_info_pred[info].update({sub_info: np.array(center_)})
        new_list.clear()

# 开始搭建训练集
# # **********************************************************************
for info in dict_info_pred:
    index = 0
    for sub_info in dict_info_pred[info]:

        img = Image.open("D:/coco/val2017/{}.jpg".format(image_id[int(info)]))
        top_left = (dict_info_pred[info][sub_info][0],
                        dict_info_pred[info][sub_info][1])
        bottom_right = (dict_info_pred[info][sub_info][2],
                            dict_info_pred[info][sub_info][3])
        cropped = img.crop((top_left[0], top_left[1],
                                bottom_right[0], bottom_right[1]))
        cropped.save("D:/couple_with_mix_box/MISTCOCO17query/{}-q-%d.jpg".format(image_id[int(info)])%(index))
        index = index + 1
