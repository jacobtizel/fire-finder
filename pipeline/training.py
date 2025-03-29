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

