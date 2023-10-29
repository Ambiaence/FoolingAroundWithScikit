import skimage as ski
from skimage.morphology import skeletonize

from skimage import graph
from skimage import data, segmentation, color, filters, io

import numpy as np
import matplotlib.pyplot as plt
from skimage.util import invert


from skimage import filters
from skimage.data import camera
from skimage.util import compare_images

image = ski.io.imread('jwords.png')[:,:,:3]
image = color.rgb2gray(image)

edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)
skeleton = skeletonize(invert(image))



fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True,
                         figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Roberts Edge Detection')

axes[1].imshow(image, cmap=plt.cm.gray)
axes[1].set_title('Original')

axes[2].imshow(skeleton, cmap=plt.cm.gray)
axes[2].set_title('skeleton')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()


