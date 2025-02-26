import os
import PIL.Image as Image
for index in os.listdir("./wsddncat12trainvaltxt/corloc"):
    print(index)
    with open(os.path.join("./wsddncat12trainvaltxt/corloc", index)) as f:
        for line in f.readlines():
            line = line.strip("\n")
            if float(line.split(" ")[1])>0.3:
                # 图像的名字
                image_name = line.split(" ")[0]
                # 打开图像
                img = Image.open("D:\VOC2007\VOCdevkit\VOC2007\JPEGImages\{}.jpg".format(image_name))
                x1 = round(float(line.split(" ")[2]))
                y1 = round(float(line.split(" ")[3]))
                x2 = round(float(line.split(" ")[4]))
                y2 = round(float(line.split(" ")[5]))
                class_info = index.split("_")[3].split(".")[0]
                cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
                cropped.save("./wsddncat12trainvaltxt/wsddncattrainvalq/{}-q-{}-{}-{}-{}-{}-{}.jpg".format(image_name, 0,
                                                                          x1, y1, x2, y2, class_info))
