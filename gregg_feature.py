from skimage import data
from skimage import data, segmentation, color, filters, io
import skimage as ski
from skimage import transform
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

img1 = ski.io.imread('gregabc.png')[:,:,:3]
img1 = color.rgb2gray(img1)

img2 = ski.io.imread('gregword.png')[:,:,:3]
img2 = color.rgb2gray(img2)


descriptor_extractor = ORB(n_keypoints=10)
descriptor_extractor.detect_and_extract(img2)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img1)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12)
ax[0].axis('off')
ax[0].set_title("Original Image vs. Transformed Image")

plt.show()