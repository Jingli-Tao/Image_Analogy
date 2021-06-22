import cv2
import math
import numpy as np

def compute_L(img_A, img_B):
    """
    To compute the number of Gaussian pyramids L:
        1. Compare the dimensions of image A and B (A and A' have the same dimensions and thus only
        compare A and B). The number of levels is decided by the one who has smaller dimension.
        2. For a given level l, the image at level l-1 has half as many pixels as the image at level
        l in each dimension. 
        3. The minimum dimension is 5*5.
    """
    min_dim = min(min(img_A.shape[0:2]), min(img_B.shape[0:2]))
    return int(math.log2(min_dim / 5))

def gaussPyramid(img, L):
    """
    Create guassian pyramid: level 0 -> L and coarse -> fine.
    Last pyramid layer is the original image.
    """
    gaussPyr = [img.astype('float64')]
    if L == 0: return gaussPyr
    for l in range(L):
        gaussPyr.append(cv2.pyrDown(gaussPyr[-1]))
    
    return gaussPyr[::-1] # coarsest to finest

def gaussPyramid_init(img_shape, L):
    """
    Initialize the gaussian pyramid for B_prime.
    """
    gaussPyr = [np.full(img_shape, np.nan)]
    if L == 0: return gaussPyr
    for i in range(L):
        h, w = gaussPyr[-1].shape
        gaussPyr.append(np.full((h // 2, w // 2), np.nan))
    
    return gaussPyr[::-1] #coarsest to finest 