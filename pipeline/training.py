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
from Preprocessing import *
from featureExtraction import *


## balancing the data
def prepare_data(df, fire_ratio=1.0):
    fire = df[df['label'] == 1]
    nonfire = df[df['label'] == 0]

    target_n = int(len(fire) * fire_ratio)
    nonfire = resample(nonfire, replace=False, n_samples=target_n, random_state=42)

    balanced = pd.concat([fire, nonfire]).sample(frac=1, random_state=42).reset_index(drop=True) #shuffle

    X = balanced.drop(columns=['label'])
    y = balanced['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler() #scale
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scaled, y_train, X_test_scaled, y_test



def generate_label_dataset(fire_dir, nonfire_dir, mask_dir, gabor_kSize = None,resize_fn=True, g_blur_fn=False, b_blur_fn=False, features='both'):
    """
    Parameters:
    - fire_dir, nonfire_dir: directories containing fire and non-fire images
    - mask_dir: directory containing masks for fire images
    - resize_fn: whether to resize images using resize_image()
    - blur_fn: whether to apply Gaussian blur
    - features: 'color', 'gabor', or 'both'
    """
    assert features in ['color', 'gabor', 'both'], "features must be one of ['color', 'gabor', 'both']"

    data = []
    assert gabor_kSize is not None, "You must explicitly set gabor_kSize"
    kernels = get_gabor_kernels(ksize=gabor_kSize)

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
        if g_blur_fn:
            g_blur_img = apply_gaussian_blur(img)
        else:
            g_blur_img = img
        if b_blur_fn:
            b_blur_img = apply_bilateral_filter(img)# use bilateral to preserve edge for gabor
        else:
            b_blur_img= img
            
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        col_channels = extract_col_features(g_blur_img, mask) if features in ['color', 'both'] else None
        gabors_feat = extract_gabor_features(b_blur_img, mask, kernels) if features in ['gabor', 'both'] else None

        if features == 'color':
            combined_feature = col_channels
        elif features == 'gabor':
            combined_feature = gabors_feat
        else:
            assert col_channels.shape[0] == gabors_feat.shape[0], "Mismatch in pixel count between color and Gabor"
            label = col_channels['label']
            col_channels = col_channels.drop(columns=['label'])
            combined_feature = pd.concat([col_channels, gabors_feat], axis=1)
            combined_feature['label'] = label

        data.append(combined_feature)

    nonfire_paths = sorted(glob(os.path.join(nonfire_dir, '*.jpg')) + glob(os.path.join(nonfire_dir, '*.png')))
    for path in tqdm(nonfire_paths, desc='Non-fire images'):
        img = cv2.imread(path)
        if img is None:
            continue

        if resize_fn:
            img = resize_image(img)
            mask = resize_image(mask)
        if g_blur_fn:
            g_blur_img = apply_gaussian_blur(img)
        else:
            g_blur_img = img
        if b_blur_fn:
            b_blur_img = apply_bilateral_filter(img)# use bilateral to preserve edge for gabor
        else:
            b_blur_img= img

        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        col_channels = extract_col_features(g_blur_img, mask) if features in ['color', 'both'] else None
        gabors_feat = extract_gabor_features(b_blur_img, mask, kernels) if features in ['gabor', 'both'] else None

        if features == 'color':
            combined_feature = col_channels
        elif features == 'gabor':
            combined_feature = gabors_feat
        else:
            assert col_channels.shape[0] == gabors_feat.shape[0], "Mismatch in pixel count between color and Gabor"
            label = col_channels['label']
            col_channels = col_channels.drop(columns=['label'])
            combined_feature = pd.concat([col_channels, gabors_feat], axis=1)
            combined_feature['label'] = label

        data.append(combined_feature)

    return pd.concat(data, ignore_index=True)



def prepare_data(df, fire_ratio=1.0):
    """
    pandas dataframe with labels, undersample the nonfire pixel to match ratio
    """
    fire = df[df['label'] == 1]
    nonfire = df[df['label'] == 0]

    target_n = int(len(fire) * fire_ratio)
    nonfire = resample(nonfire, replace=False, n_samples=target_n, random_state=42)

    balanced = pd.concat([fire, nonfire]).sample(frac=1, random_state=42).reset_index(drop=True) #shuffle

    X = balanced.drop(columns=['label'])
    y = balanced['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler() #scale
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scaled, y_train, X_test_scaled, y_test

def compute_class_based_rates(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)
    return {
        "TPR (Recall)": round(tpr*100,2),
        "FPR": round(fpr*100, 2),
        "TNR (Specificity)": round(tnr*100, 2),
        "FNR": round(fnr*100, 2)
    }
