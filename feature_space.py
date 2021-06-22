import numpy as np
import cv2
from scipy import signal

def createGaussianWeight(windowSize, sigma):
    profile = np.array(signal.gaussian(windowSize, sigma).reshape([-1, 1]))
    return profile.dot(profile.transpose())

def createCoarseWeight():
    """
    Weighting function of coarse layer.
    """
    coarseWeight = createGaussianWeight(3, .5).reshape([-1, 1])
    return coarseWeight / np.sum(coarseWeight)

def createFineWeight():
    """
    Weighting function of fine layer.
    """
    fineWeight = createGaussianWeight(5, .5).reshape([-1, 1])
    return fineWeight / np.sum(fineWeight)

def feature_vector_onePyr(gaussPyr, p, l, isPrime):
    """
    Get the feature vector in one pyramid.
    Input:
        gaussPyr: guassian pyramid
        p: pixel location
        l: level in the pyramid
        isPrime: whether the image is A' or B'.
    Output:
        feature: feature vector of [34,1] if isPrime = False or [21,1] if isPrime = True
    """  
    # Initialize as Nan for layer l and l-1 
    patch_l = np.full((5, 5), np.nan)
    patch_lminus1 = np.full((3, 3), np.nan)

    i, j = p
    h, w = gaussPyr[l].shape
    left, right, top, bottom = j - 2, j + 3, i - 2, i + 3
    left_bound, right_bound, top_bound, bottom_bound = max(0, left), min(w, right), max(0, top), min(h, bottom)
    patch_l[(top_bound - top):(5 - (bottom - bottom_bound)), (left_bound - left):(5 - (right - right_bound))] = gaussPyr[l][top_bound:bottom_bound, left_bound:right_bound]

    if l > 0:
        i, j = i // 2, j // 2
        h, w = gaussPyr[l-1].shape
        left, right, top, bottom = j - 1, j + 2, i - 1, i + 2
        left_bound, right_bound, top_bound, bottom_bound = max(0, left), min(w, right), max(0, top), min(h, bottom)
        patch_lminus1[(top_bound - top):(3 - (bottom - bottom_bound)), (left_bound - left):(3 - (right - right_bound))]=gaussPyr[l-1][top_bound:bottom_bound, left_bound:right_bound]
    
    # Concatenate all the features 3*3 and 5*5 to [34,1]
    feature = np.vstack([patch_lminus1.reshape(-1, 1), patch_l.reshape(-1, 1)])

    # If the feature is in A' or B' pyramid, remove the last half of the feature since it is not painted yet on B'
    if isPrime:
        feature = feature[:-13, :]
    return feature

def feature_vector_normalize(feature_vec, coarse_weight, fine_weight, isPrime):
    """
    Normalize the feature vector using the coarse weight and fine Weight.
    """
    weight_vec = np.vstack([coarse_weight.reshape([-1, 1]), fine_weight.reshape([-1, 1])])
    if isPrime:
        return feature_vec * weight_vec[:-13,:]
    else:
        return feature_vec * weight_vec

def feature_vector_q(Fb, gaussPyr_pb, q, l):
    """
    Update the feature vector Fb in gaussPyr_pb[l].
    """
    i, j = q
    h, w = gaussPyr_pb[l].shape
    pfb = Fb[:, i * w + j]

    # Padding if it is around the border
    if (i < 2) or (j < 2) or (i > h - 3) or (j > w - 3):
        layer_pad = cv2.copyMakeBorder(gaussPyr_pb[l], 2, 2, 2, 2, cv2.BORDER_REFLECT_101)
        patch = layer_pad[i:i+5, j:j+5]
    else:
        patch = gaussPyr_pb[l][i-2:i+3, j-2:j+3]
    
    # Reshape the patch to [25,1] and remove the last 13 features
    patch_lin = patch.reshape([25, 1])
    patch_lin_crop = patch_lin[0:12]
    pfb[34+9:34+9+12] = patch_lin_crop[:, 0]
    return pfb

def feature_space_onePyr_slowversion(gaussPyr, l, isPrime):
    """
    Create the feature space of one pyramid without vectorization.
    """
    h, w = gaussPyr[l].shape
    if isPrime:
        featureSpace = np.zeros([21, h * w])
    else:
        featureSpace = np.zeros([34, h * w])
    for i in range(h):
        for j in range(w):
            featureSpace[:, i * w + j] = feature_vector_onePyr(gaussPyr, [i, j], l, isPrime)[:, 0]
    return featureSpace

def feature_space_onePyr(gaussPyr, l, isPrime):
    """
    Create feature space of one pyramid with vectorization: 20 times speed up!
    """
    h, w = gaussPyr[l].shape
    feature_lminus1 = np.full((9, h * w), np.nan)
    feature_l = np.full((25, h * w), np.nan)

    # Create feature space for feature_lminus1
    if l > 0:
        # Double the size of gaussPyr[l] and pad the border of 2 on four sides
        coarseLayer = gaussPyr[l-1]
        coarseLayer_resized = cv2.resize(coarseLayer, (w, h))
        border = 2
        coarseLayer_resized = cv2.copyMakeBorder(coarseLayer_resized, border, border, border, border, cv2.BORDER_REFLECT_101)

        coarsefeatureList = []
        for i in range(0, 5, 2):
            for j in range(0, 5, 2):
                coarsefeatureList.append(coarseLayer_resized[i:h+i, j:w+j].reshape([1, -1]))
        feature_lminus1 = np.vstack(coarsefeatureList[::-1])

    # Create feature space for feature_l
    fineLayer = gaussPyr[l]
    border = 2
    fineLayer_border = cv2.copyMakeBorder(fineLayer, border, border, border, border, cv2.BORDER_REFLECT_101)
    finefeatureList = []
    for i in range(0, 5):
        for j in range(0, 5):
            finefeatureList.append(fineLayer_border[i:h+i, j:w+j].reshape([1, -1]))
    feature_l = np.vstack(finefeatureList[::-1])

    # Remove the last 13 features if it is A' or B'
    if isPrime:
        feature_l = feature_l[:-13, :]
   
    return np.vstack([feature_lminus1, feature_l])

def feature_space_concat(gaussPyr, gaussPyr_p, l):
    """
    Generate the feature space for (A, A') and (B, B').
    """
    feature_o = feature_space_onePyr(gaussPyr, l, False)
    feature_p = feature_space_onePyr(gaussPyr_p, l, True)
    return np.vstack([feature_o, feature_p])