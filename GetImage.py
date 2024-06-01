import cv2
import numpy as np
import os

class GetImage:
    def __init__(self, data):
        self.samples = data.values.tolist()

    def getImage(self):
        self.img_dresses = os.listdir("./samplle/dresses").sort(key=lambda x: int(x.replace("3","").split('.')[0]))
        self.img_pants = os.listdir("./samplle/pants")
        self.img_jackets = os.listdir("./samplle/jacket")
        self.img_test = os.listdir("./samplle/test")

    def readImage(self):
        samples_use = cv2.imread(pic,1) for pic in
        img = cv2.imread("./avatar.jpeg", 1)  # RGA
        print(img)  # 打印图像信息
        print(np.shape(img))  # 维度