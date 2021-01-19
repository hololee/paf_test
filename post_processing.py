import sys
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt

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

fn_im = "/data1/LJH/paf_test/train_cnn/normal/normal_0023.png"
fn_anno = "/data1/LJH/paf_test/train_result/leaf1_out.png"
fn_output = "/data1/LJH/paf_test/crf_result/output.png"
#
# fn_im = "/data1/LJH/cvpppnet/crf_test/plant001_rgb.png"
# fn_anno = "/data1/LJH/cvpppnet/crf_test/output_000.png"
# fn_output = "/data1/LJH/cvpppnet/crf_test/output.png"


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

    # # Example using the DenseCRF2D code
    # d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)
    #
    # # get unary potentials (neg log probability)
    # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
    # d.setUnaryEnergy(U)
    #
    # # This adds the color-independent term, features are the locations only.
    # d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
    #                       normalization=dcrf.NORMALIZE_SYMMETRIC)
    #
    # # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # d.addPairwiseBilateral(sxy=(15, 120), srgb=(7, 7, 7), rgbim=img,
    #                        compat=9,
    #                        kernel=dcrf.DIAG_KERNEL,
    #                        normalization=dcrf.NORMALIZE_SYMMETRIC)
else:
    print("Using generic 2D functions")

    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.3, zero_unsure=HAS_UNK)
    d.setUnaryEnergy(U)

    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=15,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(90, 90), schan=(21, 21, 21),
                                      img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=21,
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


# final_result = MAP.reshape(img.shape)
# plt.imshow(final_result)
# plt.show()




# draw
# for i in range(5):
i = 35
fig = plt.figure()
fig.set_size_inches(20, 4)  # 1800 x600

ax1 = fig.add_subplot(1, 5, 1)
ax2 = fig.add_subplot(1, 5, 2)
ax3 = fig.add_subplot(1, 5, 3)
ax4 = fig.add_subplot(1, 5, 4)
ax5 = fig.add_subplot(1, 5, 5)

ax1.set_title("origin")
ax2.set_title("output")
ax3.set_title("after CRF")
ax4.set_title("masking : over.{}".format(i))
ax5.set_title("final")

ax1.imshow(img)
ax2.imshow(anno_rgb)

final_result = MAP.reshape(img.shape)

ax3.imshow(final_result)

temp = np.zeros(final_result.shape)
temp[np.where(final_result > i)] = 255
# temp[np.where(final_result <= i + 4)] = 0

# to decide the dividing range, using iou

ax4.imshow(temp)

temp2 = np.copy(img)
temp2[np.where(final_result <= i)] = 0

ax5.imshow(temp2)

# save image.
imwrite(fn_output, temp2)

plt.show()
