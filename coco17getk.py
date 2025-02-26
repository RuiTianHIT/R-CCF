import numpy as np
import cv2
import os
import PIL.Image as Image
import random

def mask_zhouwei_box(img, image,left_top1,right_bottom1,index, mask_size,ratio):
    # image = cv2.rectangle(
    #     image, left_top1, right_bottom1, tuple((0, 255, 255)), 2
    # )
    # zhenduiyu di yige wuti
    h, w = image.shape[0], image.shape[1]
    box_w = right_bottom1[0]-left_top1[0]
    box_h = right_bottom1[1]-left_top1[1]
    total_num0 = 0
    total_dict0 = {}
    mask_dict0 = {}
    for i in range(int(left_top1[0] - 0.2*box_w), int(right_bottom1[0] + 0.2*box_w), mask_size):
        for j in range(int(left_top1[1] - 0.2*box_h), int(right_bottom1[1] + 0.2*box_h), mask_size):
            if (i < left_top1[0] - mask_size or j < left_top1[1] - mask_size) or (i > right_bottom1[0] or j > right_bottom1[1]):
                # image = cv2.rectangle(
                #     image, tuple((i, j)), tuple((i + mask_size, j + mask_size)), tuple((0, 255, 255)), 2
                # )
                total_num0 = total_num0 + 1
                total_dict0.update({total_num0: (i, j,
                                                 i + mask_size, j + mask_size)})
    mask_num0 = int(total_num0 * ratio)
    print("zhouwei_box",total_num0)
    print("mask_num", mask_num0)

    rand_mask_list = random.sample(range(1,total_num0+1),mask_num0)
    for y in rand_mask_list:
        mask_dict0.update({y: total_dict0[y]})
    print(rand_mask_list)
    print(len(rand_mask_list))

    for key in mask_dict0:
        temp_postion = mask_dict0[key]
        # (left, upper, right, lower)
        if ((int(temp_postion[0])>0) and (int(temp_postion[1])>0)) and ((int(temp_postion[2])<w) and (int(temp_postion[3])<h)):
            cropped = img.crop((temp_postion[0], temp_postion[1],
                                temp_postion[2], temp_postion[3]))
            cropped.save("D:/couple_with_mix_box/MISTCOCO17key/{}-k-{}-{}-{}-{}-{}.jpg".format(
                                                                    image_id[int(info)],
                                                                    index,
                                                                    temp_postion[0],
                                                                    temp_postion[1],
                                                                    temp_postion[2],
                                                                    temp_postion[3]))

        # for a in range(w):
        #     for b in range(h):
        #         if (a > temp_postion[0] and a < temp_postion[2]) and (b > temp_postion[1] and b < temp_postion[3]):
        #             image[b][a] = 96
    # cv2.imshow("", image)
    # cv2.waitKey(0)
    return total_dict0, mask_dict0
# --------------------------------------------------------------------
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

for info in dict_info_pred:
    index = 0
    for sub_info in dict_info_pred[info]:
        image = cv2.imread("D:/coco/val2017/{}.jpg".format(image_id[int(info)]))
        img = Image.open("D:/coco/val2017/{}.jpg".format(image_id[int(info)]))
        left_top1 = (int(dict_info_pred[info][sub_info][0]),int(dict_info_pred[info][sub_info][1]))
        right_bottom1 = (int(dict_info_pred[info][sub_info][2]), int(dict_info_pred[info][sub_info][3]))
        total_dict, mask_dict = mask_zhouwei_box(img, image, left_top1, right_bottom1, index, mask_size=32, ratio=0.6)
        print(total_dict)
        print("***")
        # this dict is true mask position
        print(mask_dict)
        index = index + 1