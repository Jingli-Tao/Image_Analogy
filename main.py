import cv2
import numpy as np
import os
import errno

from os import path
from glob import glob
from preprocess import YIQ2BGR, preprocessImages, cropOneImage
from pyramid import gaussPyramid, gaussPyramid_init, compute_L
from feature_space import feature_space_concat
from match import MatcherInit, BestMatch

def createImageAnalogy(img_A_yiq, img_A_p_yiq, img_B_yiq, pyramid_levels, ANNType="BallTree", K=4):
    """
    Notations:
        F_A: [[concat(features_A_0, features_A'_0)], [], ], all feature vectors within some 
            neighborhood N(p) of both images A and A' at all resolution levels. The neighborhood 
            is 5*5 in a given level l and 3*3 in level l-1.
        F_B: same as above.
        ANN: a list of search objects created using F_A that will be used in Best Approximate Search
        s: data structure to track the position of the source pixel p that was copied to the target
           pixel q. s(q) = p.
    """
    F_A, F_B, ANN, s = [], [], [], []

    # Compute gaussian pyramid and features
    features_A = gaussPyramid(img_A_yiq[:, :, 0], pyramid_levels)
    features_A_p = gaussPyramid(img_A_p_yiq[:, :, 0], pyramid_levels)
    features_B = gaussPyramid(img_B_yiq[:, :, 0], pyramid_levels)
    features_B_p = gaussPyramid_init(img_B_yiq[:, :, 0].shape, pyramid_levels)
   
    # Initialize search structures
    for level in range(pyramid_levels+1):
        F_A_l = feature_space_concat(features_A, features_A_p, level)
        F_B_l = feature_space_concat(features_B, features_B_p, level)
        s_l = np.zeros((features_B_p[level].shape[0], features_B_p[level].shape[1], 2))
        ANN_l = MatcherInit(F_A_l, ANNType)
        F_A.append(F_A_l)
        F_B.append(F_B_l)
        s.append(s_l)
        ANN.append(ANN_l)

    # Create image analogy
    for level in range(pyramid_levels+1):
        print("at level: " + str(level))
        h, w = features_B[level].shape
        for i in range(h):
            for j in range(w):
                q = (i, j)
                p = BestMatch(F_A, F_B, features_A_p, features_B_p, s, q, ANN, ANNType, level, K)
                features_B_p[level][q] = features_A_p[level][p[0], p[1]]
                s[level][i, j, :] = p
        if level < pyramid_levels:
            F_B[level+1] = feature_space_concat(features_B, features_B_p, level+1)
    
    h, w = s[level].shape[0:2]
    empty_yiq = np.zeros((h, w, 3))
    empty_yiq[:,:,0] = features_B_p[level]
    empty_yiq[:, :, 1:3] = img_B_yiq[:, :, 1:3]

    img_B_p = YIQ2BGR(empty_yiq)
    
    return img_B_p

def collect_files(prefix, extension_list):
    filenames = sum(map(glob, [prefix + ext for ext in extension_list]), [])
    return filenames

def main():
    input_folder = "images/input/"
    out_folder = "images/output/"
    img_exts = set(["png", "jpeg", "jpg", "gif", "tiff", "tif", "raw", "bmp"])

    subfolders = os.walk(input_folder)
    next(subfolders)  # skip the root input folder
    for dirpath, dirnames, fnames in subfolders:

        image_dir = os.path.split(dirpath)[-1]
        output_dir = os.path.join(out_folder, image_dir)

        print("Processing files in " + image_dir + " folder...")

        A_names = collect_files(os.path.join(dirpath, '*A.'), img_exts)
        A_prime_names = collect_files(os.path.join(dirpath, '*A_prime.'), img_exts)
        B_names = collect_files(os.path.join(dirpath, '*B.'), img_exts)

        if not len(A_names) == len(A_prime_names) == len(B_names) == 1:
            print("Cannot proceed. There can only be one A, A_prime, and B image in each input folder.")
            continue

        img_A = cv2.imread(A_names[0], cv2.IMREAD_COLOR)
        img_A_prime = cv2.imread(A_prime_names[0], cv2.IMREAD_COLOR)
        img_B = cv2.imread(B_names[0], cv2.IMREAD_COLOR)

        levels = compute_L(img_A, img_B)
        img_A_yiq, img_A_p_yiq, img_B_yiq = preprocessImages (img_A, img_A_prime, img_B, levels)
        img_B_prime = createImageAnalogy(img_A_yiq, img_A_p_yiq, img_B_yiq, levels, K=1)

        try:
            os.makedirs(output_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        cv2.imwrite(path.join(out_folder, "B_prime.jpg"), cropOneImage(img_B_prime, img_B.shape))

if __name__ == "__main__":
    main()