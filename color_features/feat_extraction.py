import cv2
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tqdm import tqdm
from scipy.stats import ks_2samp, ttest_ind
import seaborn as sns

from GaborFilters.gabor import apply_gabor_filters, get_diverse_gabor_kernels, get_gabor_kernels
from pipeline.Preprocessing import apply_gaussian_blur, resize_image





# function to extract color features for image and mask: return  dataframe of #pixels x 12 color channels +1 (label)
def extract_col_features(img_bgr, mask):
    # Convert color spaces
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    
    # Color features
    B, G, R = cv2.split(img_bgr)
    H, S, V = cv2.split(hsv)
    L, a, b = cv2.split(lab)
    Y, Cr, Cb = cv2.split(ycrcb)

    
    # Flatten all features
    flat_features = {
        'R': R.flatten(), 'G': G.flatten(), 'B': B.flatten(),
        'H': H.flatten(), 'S': S.flatten(), 'V': V.flatten(),
        'L': L.flatten(), 'a': a.flatten(), 'b': b.flatten(),
        'Y': Y.flatten(), 'Cr': Cr.flatten(), 'Cb': Cb.flatten(),
        'label': (mask.flatten() > 0).astype(int)
    }

    return pd.DataFrame(flat_features)


# function to extract features from image and mask: this extract both color and gabor
def extract_features(img_bgr, mask, kernels):
    # Convert color spaces
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    
    # Color features
    B, G, R = cv2.split(img_bgr)
    H, S, V = cv2.split(hsv)
    L, a, b = cv2.split(lab)
    Y, Cr, Cb = cv2.split(ycrcb)

    # Grayscale version for Gabor
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Gabor filters
    kernels = get_gabor_kernels()
    gabor_responses = apply_gabor_filters(gray, kernels)

    # Flatten all features
    flat_features = {
        'R': R.flatten(), 'G': G.flatten(), 'B': B.flatten(),
        'H': H.flatten(), 'S': S.flatten(), 'V': V.flatten(),
        'L': L.flatten(), 'a': a.flatten(), 'b': b.flatten(),
        'Y': Y.flatten(), 'Cr': Cr.flatten(), 'Cb': Cb.flatten(),
        'label': (mask.flatten() > 0).astype(int)
    }

    # Add Gabor responses
    for i, gabor in enumerate(gabor_responses):
        flat_features[f'gabor_{i}'] = gabor.flatten()

    return pd.DataFrame(flat_features)


# to generate a labeled dataset for each pixel based on the ground truth mask, do the bluring and resizing if needed
def generate_col_dataset(fire_dir, nonfire_dir, mask_dir, resize_fn=True, blur_fn=True):
    data = []

    fire_paths = sorted(glob(os.path.join(fire_dir, '*.jpg')) + glob(os.path.join(fire_dir, '*.png')))
    for path in tqdm(fire_paths, desc='Fire images'):
        name = os.path.basename(path)
        base = os.path.splitext(name)[0]
        mask_path = os.path.join(mask_dir, f"{base}_gt.png")

        img = cv2.imread(path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        if resize_fn:
            img = resize_image(img)
            mask = resize_image(mask)
        if blur_fn:
            img = apply_gaussian_blur(img)

        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        df = extract_col_features(img, mask)
        data.append(df)

    nonfire_paths = sorted(glob(os.path.join(nonfire_dir, '*.jpg')) + glob(os.path.join(nonfire_dir, '*.png')))
    for path in tqdm(nonfire_paths, desc='Non-fire images'):
        img = cv2.imread(path)
        if img is None:
            continue

        if resize_fn:
            img = resize_image(img)
        if blur_fn:
            img = apply_gaussian_blur(img)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        df = extract_col_features(img, mask)
        data.append(df)

    return pd.concat(data, ignore_index=True)



## gabor features
def generate_gabor_dataset(fire_dir, nonfire_dir, mask_dir, selected_channels=['Cb'],
                            resize_fn=True, blur_fn=False):
    """
    Generate a dataset of Gabor features computed from selected color channels for fire and non-fire images.
    
    Args:
        fire_dir (str): Directory of fire images
        nonfire_dir (str): Directory of non-fire images
        mask_dir (str): Directory of fire masks
        selected_channels (list): List of color channels to apply Gabor filters on
        resize_fn (bool): Whether to resize images using resize_image()
        blur_fn (bool): Whether to apply Gaussian blur using apply_gaussian_blur()

    Returns:
        pd.DataFrame: Dataset with Gabor features and fire labels
    """
    data = []
    kernels = get_diverse_gabor_kernels()  # 144 filters or whatever you define
    fire_paths = sorted(glob(os.path.join(fire_dir, '*.jpg')) + glob(os.path.join(fire_dir, '*.png')))

    for path in tqdm(fire_paths, desc='Fire images'):
        name = os.path.basename(path)
        base = os.path.splitext(name)[0]
        mask_path = os.path.join(mask_dir, f"{base}_gt.png")

        img = cv2.imread(path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        if resize_fn:
            img = resize_image(img)
            mask = resize_image(mask)
        if blur_fn:
            img = apply_gaussian_blur(img)

        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        df = extract_gabor_features(img, mask, selected_channels, kernels)
        data.append(df)

    # Non-fire images
    nonfire_paths = sorted(glob(os.path.join(nonfire_dir, '*.jpg')) + glob(os.path.join(nonfire_dir, '*.png')))

    for path in tqdm(nonfire_paths, desc='Non-fire images'):
        img = cv2.imread(path)
        if img is None:
            continue

        if resize_fn:
            img = resize_image(img)
        if blur_fn:
            img = apply_gaussian_blur(img)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)  # no fire
        df = extract_gabor_features(img, mask, selected_channels, kernels)
        data.append(df)

    return pd.concat(data, ignore_index=True)


def extract_gabor_features(img_bgr, mask, selected_channels, kernels):
    """
    Extract Gabor features from selected color channels for each pixel, along with label.
    """
    # Convert to color spaces
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    color_dict = {
        'Cb': img_ycrcb[:, :, 2]
    }

    features = []
    labels = []

    for ch in selected_channels:
        if ch not in color_dict:
            continue
        channel_img = color_dict[ch]
        for i, kernel in enumerate(kernels):
            filtered = cv2.filter2D(channel_img, cv2.CV_32F, kernel)
            features.append(filtered.reshape(-1))

    features = np.array(features).T  # shape: (num_pixels, num_features)
    labels = (mask.flatten() > 0).astype(int)

    df = pd.DataFrame(features, columns=[
        f'gabor_{ch}_{i}' for ch in selected_channels for i in range(len(kernels))
    ])
    df['label'] = labels
    return df
