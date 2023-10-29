from scipy.cluster.vq import kmeans2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage import data
from skimage import color
from skimage.util.shape import view_as_windows
from skimage.util import montage

patch_shape = 8, 8
n_filters = 49

astro = color.rgb2gray(data.astronaut())

print("Atroshape: ", astro.shape)
# -- filterbank1 on original image
patches1 = view_as_windows(astro, patch_shape)
print("Inital patches1: ", patches1.shape)
patches1 = patches1.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
print("Second patches1: ", patches1.shape)
fb1, _ = kmeans2(patches1, n_filters, minit='points')
print("fb1: ", fb1.shape)
fb1 = fb1.reshape((-1,) + patch_shape)
print("fb1: ", fb1.shape)
fb1_montage = montage(fb1, rescale_intensity=True)
breakpoint()

# -- filterbank2 LGN-like image
astro_dog = ndi.gaussian_filter(astro, .5) - ndi.gaussian_filter(astro, 1)
patches2 = view_as_windows(astro_dog, patch_shape)
# I think this just orders it in one dimension
patches2 = patches2.reshape(-1, patch_shape[0] * patch_shape[1])[::8]  
fb2, _ = kmeans2(patches2, n_filters, minit='points')
fb2 = fb2.reshape((-1,) + patch_shape)
fb2_montage = montage(fb2, rescale_intensity=True)

# -- plotting
fig, axes = plt.subplots(2, 2, figsize=(7, 6))
ax = axes.ravel()

ax[0].imshow(astro, cmap=plt.cm.gray)
ax[0].set_title("Image (original)")

ax[1].imshow(fb1_montage, cmap=plt.cm.gray)
ax[1].set_title("K-means filterbank (codebook)\non original image")

ax[2].imshow(astro_dog, cmap=plt.cm.gray)
ax[2].set_title("Image (LGN-like DoG)")

ax[3].imshow(fb2_montage, cmap=plt.cm.gray)
ax[3].set_title("K-means filterbank (codebook)\non LGN-like DoG image")

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()
plt.show()