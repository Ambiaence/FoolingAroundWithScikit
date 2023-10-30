import copy
from skimage.morphology import skeletonize
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import data
import skimage as ski
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage import data, segmentation, color, filters, io
# Invert the horse image
image = ski.io.imread('eword.png')[:,:,:3]
image = color.rgb2gray(image)
image = invert(image)
#image = rescale(image, (3, 1))

# perform skeletonization
skeleton = skeletonize(image)

def surrounding_pixels(pixel):
    r, c = pixel
    pixels = list()
    for i in range(-1, 2):
        for j in range(-1,2):
            if i == 0 and j == 0:
                continue

            if r + i < 0 or c + j < 0:
                continue

            pixels.append(tuple([r + i, c + j]))
    return pixels

traversed = set()
shape_lookup = dict()
shape_number = 0

for r in range(skeleton.shape[0]):
    for c in range(skeleton.shape[1]):
        starting_pixel = (r,c)
        if starting_pixel in traversed:
            continue

        if skeleton[r][c] == False:
            traversed.add(starting_pixel)
            continue

        shape_number = shape_number + 1
        to_search = set()
        traversed.add(starting_pixel)
        to_search.add(starting_pixel)
        shape_lookup[starting_pixel] = shape_number

        while to_search:
            focus_pixel = to_search.pop()
            for pixel in surrounding_pixels(focus_pixel):
                i, j = pixel
                if pixel in traversed:
                    continue
                traversed.add(pixel)
                if skeleton[i][j] == True:
                    to_search.add(pixel)
                    shape_lookup[pixel] = shape_number
    

shapes = list()
for shape_number in set(shape_lookup.values()):
    this_shape = copy.deepcopy(skeleton)
    for pixel, pixel_parent in shape_lookup.items():
        i, j = pixel
        if shape_number != pixel_parent:
            this_shape[i][j] = False 
    
    shapes.append(this_shape)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)


fig.tight_layout()
breakpoint()
plt.show()

breakpoint()

for shape in shapes:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                            sharex=True, sharey=True)

    ax = axes.ravel()

    ax[1].imshow(shape, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    fig.tight_layout()
    plt.show()
    breakpoint()