import numpy as np
import cv2

def extract_color_features(imageRGB):
    """
    Given RGB image
    Extracts pixel-level top 6 color features: H, R, S, V, Cr, a (by PCA analysis)
    Returns (pixels, 6) feature matrix.
    """
    img_hsv = cv.cvtColor(imageRGB, cv.COLOR_RGB2HSV)
    img_ycrcb = cv.cvtColor(imageRGB, cv.COLOR_RGB2YCrCb)
    img_lab = cv.cvtColor(imageRGB, cv.COLOR_RGB2Lab)

    # Flatten each channel to create (pixels, 1)
    H = img_hsv[:, :, 0].flatten().reshape(-1, 1)
    S = img_hsv[:, :, 1].flatten().reshape(-1, 1)
    V = img_hsv[:, :, 2].flatten().reshape(-1, 1)
    R = imageRGB[:, :, 0].flatten().reshape(-1, 1)
    Cr = img_ycrcb[:, :, 1].flatten().reshape(-1, 1)
    a = img_lab[:, :, 1].flatten().reshape(-1, 1)

    # Stack all selected channels together
    color_features = np.hstack([H, R, S, V, Cr, a])
    return color_features
