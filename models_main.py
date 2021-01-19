import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import method
from batch_norm import BatchNorm
from placeHolders import placeHolders

from DataGen import DataGen

import config_etc

import scipy.misc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_shape(text, input):
    sess = tf.InteractiveSession()

    print("{}shape : {}".format(text, sess.run(tf.shape(input))))
    sess.close()


dataG = DataGen()

# TODO loaded images numpy array.
rgb_images = np.array(dataG.load_images())
fg_images = np.array(dataG.load_labels())

# TODO: change 0 value to 1, becuase pydefcrf do not use 0
fg_images = np.where(fg_images == 0, 32, fg_images)

# reshape
fg_images = np.reshape(fg_images, [fg_images.shape[0], fg_images.shape[1], fg_images.shape[2], 1])

print("rgb_images : " + str(np.shape(rgb_images)))
print("fg_images : " + str(np.shape(fg_images)))

# index = np.random.randint(0, len(rgb_images))
#
# plt.imshow(rgb_images[index])
# plt.show()
# plt.imshow(fg_images[index])
# plt.show()


########### holders

ph = placeHolders(input_images=rgb_images, input_labels=fg_images)

########### layer
# first step
# get_shape("input data: {}", input_data)
gen_convolution = method.layers(method.TYPE_NORMAL, ph.input_data, 64, "layer1", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)

gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 64, "layer2_pooling", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling={'size': 2, 'stride': 2})

# step 2
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 128, "layer3", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 128, "layer3_1", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_ATROUS, gen_convolution, 128, "layer4_pooling", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling={'size': 2, 'stride': 2})

# step 3
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 256, "layer5", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 256, "layer6", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 256, "layer6_1", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 256, "layer7_pooling", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling={'size': 2, 'stride': 2})

# step 3
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 512, "layer8", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 512, "layer9", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_ATROUS, gen_convolution, 512, "layer10_atrous", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 512, "layer11", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 512, "layer12", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)
gen_convolution = method.layers(method.TYPE_ATROUS, gen_convolution, 512, "layer13_atrous", method.FUNC_RELU,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=True), pooling=None)

# for target one
gen_convolution = method.layers(method.TYPE_NORMAL, gen_convolution, 1, "layer13_finish", method.NONE,
                                BatchNorm(is_train=ph.is_train, use_batch_norm=False), pooling=None)

# bi interpolation, to original size.
h, w = dataG.getImageSize()
gen_convolution = method.bi_linear_interpolation(gen_convolution, original_map_size=(h, w))

# predict
predict_images = gen_convolution
# loss
# (batch_size, num_classes)
# num_classes = 2 : background, plant

flat_logits = tf.reshape(tensor=predict_images, shape=(-1, 2))
flat_labels = tf.reshape(tensor=ph.ground_truth, shape=(-1, 2))

# more than two classes, use soft_max_cross_entropy.
# less than two classes, use sigmoid_cross_entropy.
cross_entropies = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                        labels=flat_labels))

optimizer = tf.train.AdamOptimizer(learning_rate=ph.learning_rate).minimize(cross_entropies)

# train
BATCH_COUNT = dataG.getTotalNumber() // config_etc.BATCH_SIZE
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(config_etc.TOTAL_EPOCH):
        print("======= current epoch  : {} ======".format(epoch + 1))

        for batch_count in range(BATCH_COUNT):

            # get source batch
            batch_x, batch_y = dataG.next_batch(total_images=rgb_images, total_labels=fg_images)

            # # TODO test
            # lo, la = sess.run([flat_logits, flat_labels], feed_dict={ph.input_data: batch_x, ph.ground_truth: batch_y,ph.is_train: True})

            learn_rate = config_etc.LEARNING_RATE

            if epoch > 20:
                learn_rate = config_etc.LEARNING_RATE_v2
            elif epoch > 30:
                learn_rate = config_etc.LEARNING_RATE_v3

            # train.
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            _, _ = sess.run([optimizer, extra_update_ops], feed_dict={ph.input_data: batch_x,
                                                                      ph.ground_truth: batch_y,
                                                                      ph.is_train: True,
                                                                      ph.learning_rate: learn_rate})

            if batch_count % 5 == 0:
                # calculate loss.
                loss = sess.run(cross_entropies, feed_dict={ph.input_data: batch_x,
                                                            ph.ground_truth: batch_y,
                                                            ph.is_train: False})

                print("train_loss : {}".format(loss))

                image_result_predict = sess.run(predict_images, feed_dict={ph.input_data: batch_x, ph.is_train: False})

                print("image_result_predict # min : {} , max : {}".format(np.amin(image_result_predict),
                                                                          np.amax(image_result_predict)))

                # a = tf.nn.softmax(image_result_predict, dim=0)
                # image_result = sess.run(a)
                # after calculating loss. adjust softmax.
                # image_result_predict = image_result_predict / np.amax(image_result_predict)

                # process crf.
                # h, w = dataG.getImageSize()
                # output_crf = method.dense_crf(img=batch_x, probs=image_result_predict, n_iters=5)

                ## TODO output crf data Nan
                fig = plt.figure()
                fig.set_size_inches(9, 4)  # 1800 x600
                ax1 = fig.add_subplot(1, 3, 1)
                ax2 = fig.add_subplot(1, 3, 2)
                ax3 = fig.add_subplot(1, 3, 3)

                ax1.imshow(batch_x[0])
                ax2.imshow(np.squeeze(batch_y[0]), cmap='jet')
                ax3.imshow(np.squeeze(image_result_predict[0]), cmap='jet')

                plt.show()

    for index in range(dataG.getTotalNumber()):
        # save image.
        total_image_result_predict = sess.run(predict_images,
                                              feed_dict={
                                                  ph.input_data: np.expand_dims(dataG.load_images()[index], axis=0),
                                                  ph.is_train: False})

        scipy.misc.imsave('/data1/LJH/paf_test/train_result/leaf{}_out.png'.format(index),
                          np.squeeze(total_image_result_predict))
