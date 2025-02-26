import numpy as np
import cv2
import os
import PIL.Image as Image

with open("./boxwsddn07trainval.txt", "r") as f:  # 打开文件
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
        img = Image.open("D:/VOC2007/VOCdevkit/VOC2007/JPEGImages/{}.jpg".format(image_id))
        if image_id != "" and image_id==temp_id:
            index = index + 1
        else:
            index = 0
        cropped = img.crop((int(x1),int(y1),int(x2),int(y2)))
        # cropped.save("./wsddnq/{}-q-{}-{}-{}-{}-{}-{}.jpg".format(image_id, index,
        #                                                             x1, y1, x2, y2, class_info))
        cropped.save("./wsddnq/{}-q-{}-{}-{}-{}-{}.jpg".format(image_id, index,
                                                                  x1, y1, x2, y2))
        temp_id = image_id
