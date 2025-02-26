import numpy as np
import cv2
import os
import PIL.Image as Image

import random



def mask_zhouwei_box(image_id, img, image,left_top1,right_bottom1,index, mask_size,ratio):
    # image = cv2.rectangle(
    #     image, left_top1, right_bottom1, tuple((255, 255, 255)), 2
    # )
    # zhenduiyu di yige wuti
    # 图像的宽度和高度 获取  原始图像的宽度和高度
    h, w = image.shape[0], image.shape[1]
    box_w = right_bottom1[0]-left_top1[0]
    box_h = right_bottom1[1]-left_top1[1]
    total_num0 = 0
    total_dict0 = {}
    mask_dict0 = {}
    for i in range(int(left_top1[0] - 0.2*box_w), int(right_bottom1[0] + 0.2*box_w), mask_size):
        for j in range(int(left_top1[1] - 0.2*box_h), int(right_bottom1[1] + 0.2*box_h), mask_size):
            # if (i < left_top1[0] - mask_size or j < left_top1[1] - mask_size) or (i > right_bottom1[0] or j > right_bottom1[1]):
            # image = cv2.rectangle(
            #          image, tuple((i, j)), tuple((i + mask_size, j + mask_size)), tuple((0, 255, 0)), 2
            # )
            total_num0 = total_num0 + 1
            total_dict0.update({total_num0: (i, j, i + mask_size, j + mask_size)})
    mask_num0 = int(total_num0 * ratio)
    print("zhouwei_box", total_num0)
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
            # image = cv2.rectangle(
            #     image, tuple((temp_postion[0], temp_postion[1])),
            #     tuple((temp_postion[2], temp_postion[3])), tuple((0, 0, 255)), 2
            # )
            # cropped.save("D:/couple_with_mix_box/wsddntrainvalk/{}-k-{}-{}-{}-{}-{}-{}.jpg".format(
            #                                                         image_id,
            #                                                         index,
            #                                                         temp_postion[0],
            #                                                         temp_postion[1],
            #                                                         temp_postion[2],
            #                                                         temp_postion[3]),
            #                                                         class_info)

            cropped.save("D:/couple_with_mix_box/wsddntrainvalk/{}-k-{}-{}-{}-{}-{}.jpg".format(
                image_id,
                index,
                temp_postion[0],
                temp_postion[1],
                temp_postion[2],
                temp_postion[3]),
                )

        # for a in range(w):
        #     for b in range(h):
        #         if (a > temp_postion[0] and a < temp_postion[2]) and (b > temp_postion[1] and b < temp_postion[3]):
        #             image[b][a] = 96

    # cv2.imshow("", image)
    # cv2.waitKey(0)
    return total_dict0, mask_dict0




with open("boxwsddn07trainval.txt", "r") as f:  # 打开文件
    index = 0
    temp_id = ""
    for line in f.readlines():
        line = line.strip("\n")
        data = line.split(",")
        image_id = data[0]
        x1 = data[1]
        y1 = data[2]
        x2 = data[3]
        y2 = data[4]
        # class_info = data[5]
        # 传入的是最有判别力区域的坐标
        left_top1 = (int(x1), int(y1))
        right_bottom1 = (int(x2), int(y2))
        if temp_id != "" and image_id == temp_id:
            index = index + 1
        else:
            index = 0

        img = Image.open("D:/VOC2007/VOCdevkit/VOC2007/JPEGImages/{}.jpg".format(image_id))
        image = cv2.imread("D:\VOC2007\VOCdevkit\VOC2007\JPEGImages/{}.jpg".format(image_id))
        total_dict, mask_dict = mask_zhouwei_box(image_id, img, image, left_top1,
                                                 right_bottom1, index,
                                                 mask_size=32, ratio=0.6)
        print(total_dict)
        print("***")
        # this dict is true mask position
        print(mask_dict)
        temp_id = image_id
