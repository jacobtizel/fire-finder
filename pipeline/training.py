import cv2
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.calibration import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from tqdm import tqdm
from scipy.stats import ks_2samp, ttest_ind
import seaborn as sns
import joblib
import time

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

def fit_linear_svc(X_train, y_train):
    """
    Fit an SVM model to the training data with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        best_model: Trained SVM model with best parameters
        best_params: Dictionary of best parameters
    """
    start_time = time.time()
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'loss': ['hinge', 'squared_hinge'],
        'penalty': ['l2'],
        'dual': [True],
    }
    
    lsvc = LinearSVC(verbose=1)
    grid_search = GridSearchCV(estimator=lsvc, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    
    print("Starting GridSearchCV for LinearSVC...")
    grid_search.fit(X_train, y_train)
    print("Finished GridSearchCV.")
    
    # best model and params
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # print them
    print("\n=== Best Parameters Found ===")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # how long did it take
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # save it
    model_dir = os.path.join('models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'linear_svc_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    return best_model, best_params