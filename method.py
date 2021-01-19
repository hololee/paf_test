import tensorflow as tf
import tensorflow as tf
import config_etc
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt

# convolution type.
TYPE_NORMAL = 'normal'
TYPE_ATROUS = 'atrous'

# activate functions.
FUNC_RELU = 'relu'
NONE = 'none'


# deeplab model layer
def layers_deeplab(type, input_map, tar_dim, name, act_func, batch_norm, pooling={'size': 2, 'stride': 2}):
    weight = tf.Variable(
        tf.random_normal([3, 3, input_map.get_shape().as_list()[3], tar_dim], stddev=config_etc.TRAIN_STDDV), name=name)
    bias = tf.Variable([0.1])

    # choose type
    if type == TYPE_NORMAL:
        conv_result = tf.nn.conv2d(input_map, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
    elif type == TYPE_ATROUS:
        conv_result = tf.nn.atrous_conv2d(input_map, weight, rate=2, padding="VALID") + bias

    # activation
    if act_func == None:
        pass
    elif act_func == FUNC_RELU:
        conv_result = tf.nn.relu(conv_result, name + FUNC_RELU)

    # batch normalization
    if batch_norm.use_batch_norm:
        # using batch normalization.
        conv_result = tf.layers.batch_normalization(conv_result, center=True, scale=True, training=batch_norm.is_train)

    # max pooling.
    if pooling != None:
        conv_result = tf.nn.max_pool(conv_result, ksize=[1, pooling['size'], pooling['size'], 1],
                                     strides=[1, pooling['stride'], pooling['stride'], 1],
                                     padding='SAME')

    # print shape of array
    print(conv_result.shape)

    return conv_result


# Enet model layer
def layer_Enet_initial(input_map, name):
    # concate 13 conv features , 3 pooling result, output has 16 dim size.
    conv_weight = tf.Variable(
        tf.random_normal(shape=[3, 3, input_map.get_shape().as_list()[3], 13], stddev=config_etc.TRAIN_STDDV),
        name=name + "_filter")
    conv_bias = tf.Variable([0.1], name=name + "_bias")
    conv_part = tf.nn.conv2d(input_map, conv_weight, strides=[1, 2, 2, 1], padding="SAME",
                             name=name + "_conv_part") + conv_bias
    pooling_part = tf.nn.max_pool(input_map, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    concat = tf.concat(values=[conv_part, pooling_part], axis=-1)

    print("layer({}) : {}".format(name, concat.get_shape()))

    return concat


def layer_enet_bottle_neck(input_map, layer_type, training, name):
    """

    :param input_map:
    :param layer_type:
        dic format:
        "ver" : 1, 2  - bottleneck version 1 or 2
        "type" : "regular", "dilated" ,"transpose_conv", "asymmetric"
        "down_sampling" : True, False

        "conv_size" : (int) size
        "dilated_rate" : (int) rate
        "asymmetric_rate" : (int) rate
        "target_dim" : (int) dim
        "projection_ratio" : (int) ratio


    :param training:
    :param name:
    :return:
    """

    if layer_type["ver"] == 1:
        drop_param = 0.01
    else:
        drop_param = 0.1

    # 1by1 conv---------------------------------------------------------------------------------------------------------
    temp_val = (lambda x: 2 if layer_type["down_sampling"] == x else 1)(1)
    weight_1 = tf.Variable(
        tf.random_normal(
            shape=[temp_val, temp_val, input_map.get_shape().as_list()[3],
                   input_map.get_shape().as_list()[3] // layer_type["projection_ratio"]]))
    weighted = tf.nn.conv2d(input_map, weight_1, strides=[1, temp_val, temp_val, 1],
                            padding="SAME",
                            name=name + "_first_1by1")
    weighted = p_relu(weighted, name + "alpha1")
    weighted = tf.layers.batch_normalization(weighted, center=True, scale=True, training=training)
    print("┌ step_1by1 : {}".format(weighted.get_shape()))
    # CONV--------------------------------------------------------------------------------------------------------------
    if layer_type["type"] == "regular":
        weight_regular = tf.Variable(tf.random_normal(
            shape=[layer_type["conv_size"], layer_type["conv_size"], weighted.get_shape().as_list()[3],
                   input_map.get_shape().as_list()[3] // layer_type["projection_ratio"]],
            stddev=config_etc.TRAIN_STDDV), name=name + "_regular")
        weighted = tf.nn.conv2d(weighted, weight_regular, strides=[1, 1, 1, 1], padding="SAME",
                                name=name + "_regular")

    # 다른 conv 방식.
    elif layer_type["type"] == "dilated":
        weight_dilated = tf.Variable(tf.random_normal(
            shape=[layer_type["conv_size"], layer_type["conv_size"], weighted.get_shape().as_list()[3],
                   weighted.get_shape().as_list()[3]], stddev=config_etc.TRAIN_STDDV), name=name + "_dilated")
        weighted = tf.nn.atrous_conv2d(weighted, weight_dilated, rate=layer_type["dilated_rate"], padding="SAME")

    elif layer_type["type"] == "transpose_conv":
        reduced_depth = input_map.get_shape().as_list()[3] // layer_type["projection_ratio"]

        weight_deconv = tf.Variable(tf.random_normal(
            shape=[layer_type["conv_size"], layer_type["conv_size"], weighted.get_shape().as_list()[3],
                   weighted.get_shape().as_list()[3]], stddev=config_etc.TRAIN_STDDV), name=name + "_transpose_conv")

        weighted = tf.nn.conv2d_transpose(value=weighted, filter=weight_deconv,
                                          output_shape=[weighted.get_shape().as_list()[0],
                                                        weighted.get_shape().as_list()[1] * 2,
                                                        weighted.get_shape().as_list()[2] * 2, reduced_depth],
                                          strides=[1, 2, 2, 1], padding="SAME")


    elif layer_type["type"] == "asymmetric":
        asymmetric_w_1 = tf.Variable(
            tf.random_normal(
                [layer_type["asymmetric_rate"], 1, weighted.get_shape().as_list()[3],
                 weighted.get_shape().as_list()[3]],
                stddev=config_etc.TRAIN_STDDV), name=name + "_asymmetric1")
        weighted = tf.nn.conv2d(weighted, asymmetric_w_1, strides=[1, 1, 1, 1], padding="SAME")
        asymmetric_w_2 = tf.Variable(
            tf.random_normal(
                [1, layer_type["asymmetric_rate"], weighted.get_shape().as_list()[3],
                 weighted.get_shape().as_list()[3]],
                stddev=config_etc.TRAIN_STDDV), name=name + "_asymmetric2")
        weighted = tf.nn.conv2d(weighted, asymmetric_w_2, strides=[1, 1, 1, 1], padding="SAME")

    print("┌ step_conv_{} : {}".format(layer_type["type"], weighted.get_shape()))
    weighted = p_relu(weighted, name=name + "alpha2")
    weighted = tf.layers.batch_normalization(weighted, center=True, scale=True, training=training)

    # 1by1 conv---------------------------------------------------------------------------------------------------------
    weight_2 = tf.Variable(
        tf.random_normal(shape=[1, 1, weighted.get_shape().as_list()[3],
                                layer_type["target_dim"]]))
    weighted = tf.nn.conv2d(weighted, weight_2, strides=[1, 1, 1, 1],
                            padding="SAME",
                            name=name + "_second_1by1")
    print("┌ step_1by1 : {}".format(weighted.get_shape()))
    # dropout - regulaizer ---------------------------------------------------------------------------------------------------------
    weighted = tf.layers.dropout(weighted, rate=drop_param, training=training, name=name + "_dropout")

    # down sampling -----------------------------------------------------------------------------------------------------
    if layer_type["down_sampling"]:
        max_pool = tf.nn.max_pool(input_map, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                  name=name + "_downsamp")

        if layer_type["ver"] != "full_conv":
            inputs_shape = input_map.get_shape().as_list()
            depth_to_pad = abs(inputs_shape[3] - layer_type["target_dim"])

            # padding 0 dims
            paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, depth_to_pad]])
            max_pool = tf.pad(max_pool, paddings=paddings, name=name + '_padding')

            # TODO : why using add?
            weighted = tf.add(weighted, max_pool)

            print("┌ # step(down_sampling) : {}".format(weighted.get_shape()))

    print("layer({}) : {}".format(name, weighted.get_shape()))

    return weighted


def bi_linear_interpolation(input_map, original_map_size=(530, 500)):
    conv_result = tf.image.resize_images(input_map, size=original_map_size,
                                         method=tf.image.ResizeMethod.BILINEAR)

    # print shape of array
    print(conv_result.shape)

    return conv_result


def p_relu(_x, name):
    alphas = tf.get_variable(name, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


# ====== crf ========
def apply_crf(original_image_path, output_image_path, final_result_path):
    # Get im{read,write} from somewhere.
    try:
        from cv2 import imread, imwrite
    except ImportError:
        # Note that, sadly, skimage unconditionally import scipy and matplotlib,
        # so you'll need them if you don't have OpenCV. But you probably have them.
        from skimage.io import imread, imsave

        imwrite = imsave
        # TODO: Use scipy instead.

    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

    fn_im = original_image_path
    fn_anno = output_image_path
    fn_output = final_result_path

    # fn_im = "/data1/LJH/cvpppnet/A1/plant002_rgb.png"
    # fn_anno = "/data1/LJH/cvpppnet/A1_predict/output_001.png"
    # fn_output = "/data1/LJH/cvpppnet/semantic_segmentation_usinng_crf/output.png"

    ##################################
    ### Read images and annotation ###
    ##################################
    img = imread(fn_im)

    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    anno_rgb = imread(fn_anno).astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
        print(
            "Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print(
            "If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    # else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    ###########################
    ### Setup the CRF model ###
    ###########################
    use_2d = False
    # use_2d = True
    if use_2d:
        print("Using 2D specialized functions")

        # Example using the DenseCRF2D code
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=6, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(95, 95), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        print("Using generic 2D functions")

        # Example using the DenseCRF class and the util functions
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]
    # TODO: save image

    # Just randomly manually run inference iterations
    Q, tmp1, tmp2 = d.startInference()
    for i in range(3):
        print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)

    # final result.
    final_result = MAP.reshape(img.shape)
    rt = np.zeros(final_result.shape)
    rt[np.where(final_result > 30)] = 255

    # save image.
    imwrite(fn_output, rt)

    fig = plt.figure()
    fig.set_size_inches(10, 2)  # 1800 x600

    ax1 = fig.add_subplot(1, 5, 1)
    ax2 = fig.add_subplot(1, 5, 2)
    ax3 = fig.add_subplot(1, 5, 3)
    ax4 = fig.add_subplot(1, 5, 4)
    ax5 = fig.add_subplot(1, 5, 5)

    ax1.set_title("origin")
    ax2.set_title("output")
    ax3.set_title("after CRF")
    ax4.set_title("masking : over.{}".format(30))
    ax5.set_title("final")

    ax1.imshow(img)
    ax2.imshow(anno_rgb)

    final_result = MAP.reshape(img.shape)

    ax3.imshow(final_result)

    temp = np.zeros(final_result.shape)
    temp[np.where(final_result > 30)] = 255

    # to decide the dividing range, using iou

    ax4.imshow(temp)

    temp2 = np.copy(img)
    temp2[np.where(final_result <= 30)] = 0

    ax5.imshow(temp2)

    # save image.
    imwrite(fn_output, temp2)

    plt.show()

    return rt
