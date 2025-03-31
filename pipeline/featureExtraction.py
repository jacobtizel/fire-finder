import numpy as np
import cv2 as cv
import pandas as pd

# def extract_color_features(imageRGB):
#     """
#     Given RGB image
#     Extracts pixel-level top 6 color features: H, R, S, V, Cr, a (by PCA analysis)
#     Returns (pixels, 6) feature matrix.
#     """
#     img_hsv = cv.cvtColor(imageRGB, cv.COLOR_RGB2HSV)
#     img_ycrcb = cv.cvtColor(imageRGB, cv.COLOR_RGB2YCrCb)
#     img_lab = cv.cvtColor(imageRGB, cv.COLOR_RGB2Lab)

#     # Flatten each channel to create (pixels, 1)
#     H = img_hsv[:, :, 0].flatten().reshape(-1, 1)
#     S = img_hsv[:, :, 1].flatten().reshape(-1, 1)
#     V = img_hsv[:, :, 2].flatten().reshape(-1, 1)
#     R = imageRGB[:, :, 0].flatten().reshape(-1, 1)
#     Cr = img_ycrcb[:, :, 1].flatten().reshape(-1, 1)
#     a = img_lab[:, :, 1].flatten().reshape(-1, 1)

#     # Stack all selected channels together
#     color_features = np.hstack([H, R, S, V, Cr, a])
#     return color_features


## Gabor

def get_gabor_kernels(ksize=None, sigma=1.0):
    """Generate a bank of Gabor kernels at different orientations and wavelengths"""
    #4 orientations,2 wavelengths,2 aspect ratios.
    kernels = []
    print(f"Using ksize = {ksize}")
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for lambd in [4, 8]:
            for gamma in [0.5, 1.0]:# both circular and ellipical
                kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0)
                print(f"ksize: {ksize}, kernel shape: {kernel.shape}")
                kernels.append(kernel)
    return kernels


def apply_gabor_filters(gray_img, kernels):
    """Apply a list of Gabor kernels to a grayscale image, return a list of responses"""
    responses = []
    for kernel in kernels:
        filtered = cv.filter2D(gray_img, cv.CV_32F, kernel)
        responses.append(filtered)
    return responses  # List of arrays, shape (H, W) each


# function to extract features from image and mask
def extract_col_features(img_rgb, mask):
    """
    assumes RGB image is passed, extract color features  and assign label for each pixel
    """
    # Convert color spaces
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    ycrcb = cv.cvtColor(img_rgb, cv.COLOR_RGB2YCrCb)
    
    # Color features
    R, G, B= cv.split(img_rgb)
    H, S, V = cv.split(hsv)
    L, a, b = cv.split(lab)
    Y, Cr, Cb = cv.split(ycrcb)

    
    # Flatten all features
    flat_features = {
        'R': R.flatten(), 'G': G.flatten(), 'B': B.flatten(),
        'H': H.flatten(), 'S': S.flatten(), 'V': V.flatten(),
        'L': L.flatten(), 'a': a.flatten(), 'b': b.flatten(),
        'Y': Y.flatten(), 'Cr': Cr.flatten(), 'Cb': Cb.flatten(),
        'label': (mask.flatten() > 0).astype(int)
    }

    return pd.DataFrame(flat_features)



# function to extract gabor features features from image and mask
def extract_gabor_features(img_rgb, mask, kernels):
    """
    assumes RGB after bluring
    """
    # Grayscale version for Gabor
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)

    # Apply Gabor filters
    #kernels = get_gabor_kernels()
    gabor_responses = apply_gabor_filters(gray, kernels)

    # Flatten all features
    flat_features = {'label': (mask.flatten() > 0).astype(int)}

    # Add Gabor responses
    for i, gabor in enumerate(gabor_responses):
        flat_features[f'gabor_{i}'] = gabor.flatten()

    return pd.DataFrame(flat_features)