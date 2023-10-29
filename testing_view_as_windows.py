from scipy.cluster.vq import kmeans2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage import color
from skimage.util.shape import view_as_windows
from skimage.util import montage

euclid = np.array(list(range(1,26))).reshape((5,5))
print("Shape of euclid", euclid.shape)
element = view_as_windows(euclid, (2, 2))
print("Shape of element", element.shape)
print(element)
breakpoint()