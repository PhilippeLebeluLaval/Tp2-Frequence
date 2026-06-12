import numpy as np
from scipy.ndimage import gaussian_filter

def hybrid_image(im1, im2, cutoff_low, cutoff_high):

    if im1.dtype == np.uint8:
        im1 = im1.astype(np.float32) / 255.0
    else:
        im1 = im1.astype(np.float32) / (im1.max() if im1.max() > 1 else 1.0)

    if im2.dtype == np.uint8:
        im2 = im2.astype(np.float32) / 255.0
    else:
        im2 = im2.astype(np.float32) / (im2.max() if im2.max() > 1 else 1.0)

    low_frequencies = gaussian_filter(im1, sigma=cutoff_low)

    low_im2 = gaussian_filter(im2, sigma=cutoff_high)
    high_frequencies = (im2 - low_im2) + 0.5

    hybrid = (low_frequencies + high_frequencies) / 2.0

    hybrid = np.clip(hybrid, 0, 1)
    return (hybrid * 255).astype(np.uint8)
