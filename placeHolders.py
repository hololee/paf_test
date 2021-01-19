import tensorflow as tf
import numpy as np
import config_etc


class placeHolders:

    def __init__(self, input_images, input_labels):
        self.input_data = tf.placeholder(tf.float32, [config_etc.BATCH_SIZE, np.shape(input_images)[1], np.shape(input_images)[2],
                                                      np.shape(input_images)[3]])
        self.ground_truth = tf.placeholder(tf.float32, [config_etc.BATCH_SIZE, np.shape(input_labels)[1], np.shape(input_labels)[2],
                                                        np.shape(input_labels)[3]])

        self.is_train = tf.placeholder(tf.bool)

        self.learning_rate = tf.placeholder(tf.float32)
