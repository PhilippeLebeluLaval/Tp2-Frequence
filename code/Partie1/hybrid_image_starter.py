from imageio import imread
from align_images import align_images
from crop_image import crop_image
from hybrid_image import hybrid_image
from stacks import stacks

# read images
im1 = imread('./Marilyn_Monroe.png', pilmode='L')
im2 = imread('./Albert_Einstein.png', pilmode='L')

# use this if you want to align the two images (e.g., by the eyes) and crop
# them to be of same size
im1, im2 = align_images(im1, im2)

# Choose the cutoff frequencies and compute the hybrid image (you supply
# this code)
arbitrary_value_1 = 6
arbitrary_value_2 = 3
cutoff_low = arbitrary_value_1
cutoff_high = arbitrary_value_2
im12 = hybrid_image(im1, im2, cutoff_low, cutoff_high)

# Crop resulting image (optional)
assert im12 is not None, "im12 is empty, implement hybrid_image!"
im12 = crop_image(im12)

# Compute and display Gaussian and Laplacian Stacks (you supply this code)
n = 5  # number of pyramid levels (you may use more or fewer, as needed)
stacks(im12, n)
