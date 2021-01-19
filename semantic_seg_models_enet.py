import method as md
import tensorflow as tf
import numpy as np
import os
from DataGen import DataGen
from placeHolders import placeHolders
import config_etc
import matplotlib.pyplot as plt
import scipy.misc
import datetime
import time

current_milli_time = lambda: int(round(time.time() * 1000))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

## ready dataset .
dataG = DataGen()

# loaded images numpy array.
rgb_images = np.array(dataG.load_images())
fg_images = np.array(dataG.load_labels())

# reshape
fg_images = np.reshape(fg_images, [fg_images.shape[0], fg_images.shape[1], fg_images.shape[2], 1])

print("rgb_images : " + str(np.shape(rgb_images)))
print("fg_images : " + str(np.shape(fg_images)))

# create input place hodler and apply.
ph = placeHolders(input_images=rgb_images, input_labels=fg_images)

## network set.
num_classes = 1

# ==== initial
net = md.layer_Enet_initial(ph.input_data, name="initial")
print(ph.input_data.get_shape())
# net = md.layer_Enet_initial(input, name="initial")

# ==== ver 1
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 1, "type": "regular", "down_sampling": True, "conv_size": 3,
                                                 "target_dim": 64, "projection_ratio": 4},
                                training=ph.is_train, name="bottleneck1_0")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 1, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 64, "projection_ratio": 4},
                                training=ph.is_train, name="bottleneck1_1")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 1, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 64, "projection_ratio": 4},
                                training=ph.is_train, name="bottleneck1_2")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 1, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 64, "projection_ratio": 4},
                                training=ph.is_train, name="bottleneck1_3")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 1, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 64, "projection_ratio": 4},
                                training=ph.is_train, name="bottleneck1_4")

# ==== ver 2
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "regular", "down_sampling": True, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck2_0")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck2_1")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 2},
                                training=ph.is_train,
                                name="bottleneck2_2")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "asymmetric", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "asymmetric_rate": 5},
                                training=ph.is_train,
                                name="bottleneck2_3")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 4},
                                training=ph.is_train,
                                name="bottleneck2_4")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck2_5")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 8},
                                training=ph.is_train,
                                name="bottleneck2_6")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "asymmetric", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "asymmetric_rate": 5},
                                training=ph.is_train,
                                name="bottleneck2_7")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 16},
                                training=ph.is_train,
                                name="bottleneck2_8")


#==== ver3
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck3_1")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 2},
                                training=ph.is_train,
                                name="bottleneck3_2")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "asymmetric", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "asymmetric_rate": 5},
                                training=ph.is_train,
                                name="bottleneck3_3")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 4},
                                training=ph.is_train,
                                name="bottleneck3_4")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck3_5")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 8},
                                training=ph.is_train,
                                name="bottleneck3_6")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "asymmetric", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "asymmetric_rate": 5},
                                training=ph.is_train,
                                name="bottleneck3_7")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 16},
                                training=ph.is_train,
                                name="bottleneck3_8")

# ==== ver 4
net = md.layer_enet_bottle_neck(net,
                                layer_type={"ver": 4, "type": "transpose_conv", "down_sampling": False, "conv_size": 3,
                                            "target_dim": 64, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck4_0")
net = md.layer_enet_bottle_neck(net,
                                layer_type={"ver": 4, "type": "regular", "down_sampling": False, "conv_size": 3,
                                            "target_dim": 64, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck4_1")
net = md.layer_enet_bottle_neck(net,
                                layer_type={"ver": 4, "type": "regular", "down_sampling": False, "conv_size": 3,
                                            "target_dim": 64, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck4_2")

# ==== ver 5
net = md.layer_enet_bottle_neck(net,
                                layer_type={"ver": 5, "type": "transpose_conv", "down_sampling": False, "conv_size": 3,
                                            "target_dim": 16, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck5_0")
net = md.layer_enet_bottle_neck(net,
                                layer_type={"ver": 5, "type": "regular", "down_sampling": False, "conv_size": 3,
                                            "target_dim": 16, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck5_1")

# fullconv
net = md.layer_enet_bottle_neck(net,
                                layer_type={"ver": "full_conv", "type": "transpose_conv", "down_sampling": False,
                                            "conv_size": 3,
                                            "target_dim": num_classes, "projection_ratio": 4}, training=ph.is_train,
                                name="full_conv")

# predict
predict_images = net
# loss
# (batch_size, num_classes)
# num_classes = 2 : background, plant
flat_logits = tf.reshape(tensor=predict_images, shape=(-1, 2))
flat_labels = tf.reshape(tensor=ph.ground_truth, shape=(-1, 2))

# more than two classes, use soft_max_cross_entropy.
# less than two classes, use sigmoid_cross_entropy.
cross_entropies = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                        labels=flat_labels, dim=-1))

optimizer = tf.train.AdamOptimizer(learning_rate=ph.learning_rate).minimize(cross_entropies)

# train
BATCH_COUNT = dataG.getTotalNumber() // config_etc.BATCH_SIZE
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(config_etc.TOTAL_EPOCH):
        print("======= current epoch  : {} ======".format(epoch + 1))

        learn_rate = config_etc.LEARNING_RATE
        if epoch > 65:
            learn_rate = config_etc.LEARNING_RATE_v2
        if epoch > 100:
            learn_rate = config_etc.LEARNING_RATE_v3


        for batch_count in range(BATCH_COUNT):

            # get source batch
            batch_x, batch_y = dataG.next_batch(total_images=rgb_images, total_labels=fg_images)

            # train.
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            _, _ = sess.run([optimizer, extra_update_ops], feed_dict={ph.input_data: batch_x,
                                                                      ph.ground_truth: batch_y,
                                                                      ph.is_train: True,
                                                                      ph.learning_rate: config_etc.LEARNING_RATE})

            if epoch == (config_etc.TOTAL_EPOCH - 1):
                image_result_predict = sess.run(predict_images, feed_dict={ph.input_data: batch_x, ph.is_train: False})
                for index, image in enumerate(image_result_predict):
                    # save image.
                    suffix = current_milli_time()
                    scipy.misc.imsave(
                        '/data1/LJH/paf_test/train_result_ent/leaf_in_epc{}_{}.png'.format(epoch + 1, suffix),
                        np.squeeze(batch_x[index]))
                    scipy.misc.imsave(
                        '/data1/LJH/paf_test/train_result_ent/leaf_out_epc{}_{}.png'.format(epoch + 1, suffix),
                        np.squeeze(image))

            if batch_count % 4 == 0:
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

                fig = plt.figure()
                fig.set_size_inches(9, 4)  # 1800 x600
                ax1 = fig.add_subplot(1, 3, 1)
                ax2 = fig.add_subplot(1, 3, 2)
                ax3 = fig.add_subplot(1, 3, 3)

                ax1.imshow(batch_x[0])
                ax2.imshow(np.squeeze(batch_y[0]), cmap='jet')
                ax3.imshow(np.squeeze(image_result_predict[0]), cmap='jet')

                plt.show()
