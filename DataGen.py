from scipy.misc import imread, imsave, imresize
import numpy as np
import os
import config_etc

from PIL import Image


class DataGen:

    def __init__(self):
        self.batch_flag = 0

        self.dir1 = '/data1/LJH/paf_test/train_cnn/normal'
        self.dir2 = '/data1/LJH/paf_test/train_target/normal'

        all_img_list = os.listdir(self.dir1)
        all_img_list.sort()

        all_target_list = os.listdir(self.dir2)
        all_target_list.sort()

        self.rgbs_name = []
        self.fgs_name = []

        self.rgb_images = []
        self.fg_images = []

        # slect origin images.

        for titles in all_img_list:
            print(titles + ": color image")
            # color images
            self.rgbs_name.append(titles)

        for titles in all_target_list:
            print(titles + ": foreground")
            # foregrounds
            self.fgs_name.append(titles)

        print("\n==========================================")
        # number of total images.
        self.total_number = len(self.rgbs_name)
        print("number of total data : {}".format(len(self.rgbs_name)))
        print("==========================================\n")

    def load_images(self):
        for image_names in self.rgbs_name:
            real_path = self.dir1 + "/" + image_names
            # load images.
            self.rgb_images.append(imread(real_path, mode='RGB'))

        return self.rgb_images

    def load_labels(self):
        for image_names in self.fgs_name:
            real_path = self.dir2 + "/" + image_names
            # load images.
            self.fg_images.append(imread(real_path, mode='L'))

        # change range  0 to 1
        # self.fg_images = self.fg_images / np.amax(self.fg_images)

        return self.fg_images

    # get total number.
    def getTotalNumber(self):
        return self.total_number

    # get image shpape.
    def getImageSize(self):
        # height, width.
        return np.shape(np.array(self.rgb_images))[1], np.shape(np.array(self.rgb_images))[2]

    # next batch.
    def next_batch(self, total_images, total_labels):

        batch_size = config_etc.BATCH_SIZE

        sub_batch_x, sub_batch_y = total_images[self.batch_flag: self.batch_flag + batch_size], total_labels[
                                                                                                self.batch_flag:self.batch_flag + batch_size]
        self.batch_flag = (self.batch_flag + batch_size) % len(total_images)
        return sub_batch_x, sub_batch_y
