import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

dir = '/data1/LJH/paf_test/train_target/normal'
all_img_list = os.listdir(dir)
all_img_list.sort()

for titles in all_img_list:
    img = cv2.imread('/data1/LJH/paf_test/train_target/normal/{}'.format(titles), 0)
    ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite('/data1/LJH/paf_test/train_target/normal/{}'.format(titles), thresh_img)
