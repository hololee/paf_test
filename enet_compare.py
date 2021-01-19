import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.misc

dir1 = '/data1/LJH/paf_test/train_result_ent'

all_img_list = os.listdir(dir1)
all_img_list.sort()

rgbs_name = []
fgs_name = []

rgb_images = []
fg_images = []

# slect origin images.

for titles in all_img_list:
    print(titles + ": color image")
    # color images

    if "leaf_in" in titles:
        rgbs_name.append(titles)
    else:
        fgs_name.append(titles)

for image_names in rgbs_name:
    real_path = dir1 + "/" + image_names
    # load images.
    rgb_images.append(imread(real_path, mode='RGB'))

for image_names in fgs_name:
    real_path = dir1 + "/" + image_names
    # load images.
    fg_images.append(imread(real_path, mode='L'))

for idx in range(len(rgb_images)):
    fig = plt.figure()
    fig.set_size_inches(15, 5)

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(rgb_images[idx])
    ax2.imshow(fg_images[idx])
    # masking
    ax2.imshow(fg_images[idx])

    temp = np.copy(rgb_images[idx])
    temp[np.where(fg_images[idx] <= 35)] = 0

    ax3.imshow(temp)

    plt.show()

    # save image
    fig.savefig('/data1/LJH/paf_test/enet_result/result_{}.png'.format(idx))
