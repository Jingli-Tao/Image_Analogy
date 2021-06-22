import numpy as np
import cv2

def padOneImage(img, L):
    max_factor = 2 ** L

    # Compute the total pad border in horizontal and vertial direction
    h, w = img.shape[0:2]
    wa_pad_total = max_factor * (w // max_factor + 1) - w
    ha_pad_total = max_factor * (h // max_factor + 1) - h

    # Compute border top, bottom, left, and right values.
    top, bottom = ha_pad_total // 2, ha_pad_total - ha_pad_total // 2
    left, right = wa_pad_total // 2, wa_pad_total - wa_pad_total // 2

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

def cropOneImage(img_prime, img_shape):
    h, w = img_prime.shape[0:2]
    h_t, w_t = img_shape[0:2]
    wa_pad_total = w - w_t
    ha_pad_total = h - h_t

    top, bottom = ha_pad_total // 2, ha_pad_total - ha_pad_total // 2
    left, right = wa_pad_total // 2, wa_pad_total - wa_pad_total // 2

    return img_prime[top:(h - bottom), left:(w - right), :]

def BGR2YIQ(img_bgr):
    """Convert RGB image to YIQ image.
    Input:
        img_bgr: image read with imread.
    Return:
        img_yig: YIQ image.
    """
    conversionMat = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]]) # numbers are from https://en.wikipedia.org/wiki/YIQ#From_RGB_to_YIQ
    img_rgb = img_bgr[:, :, ::-1] # convert BGR to RGB
    h, w = img_rgb.shape[0:2]
    pixels = img_rgb.reshape([-1, 3])
    img_yiq = conversionMat.dot(pixels.transpose()).transpose().reshape(h, w, 3)
    return img_yiq

def YIQ2BGR(img_yiq):
    """Convert YIQ image to RGB image.
    Input:
        img_yig: YIQ image.  
    Return:
        img_bgr: image read with imread.
    """
    conversionMat = np.array([[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]]) # numbers are from https://en.wikipedia.org/wiki/YIQ#From_YIQ_to_RGB
    h, w = img_yiq.shape[0:2]
    pixels = img_yiq.reshape([-1, 3])
  
    img_rgb = conversionMat.dot(pixels.transpose()).transpose().reshape(h, w, 3)
    img_bgr = img_rgb[:, :, ::-1] # convert RGB to BGR
    img_bgr = np.clip(img_bgr, 0, 255)
    return img_bgr.astype('uint8')

def preprocessImages(img_A, img_A_prime, img_B, L):
    A = padOneImage(img_A, L)
    A_prime = padOneImage(img_A_prime, L)
    B = padOneImage(img_B, L)

    A_yiq = BGR2YIQ(A)
    A_prime_yiq = BGR2YIQ(A_prime)
    B_yiq = BGR2YIQ(B)   

    return A_yiq, A_prime_yiq, B_yiq