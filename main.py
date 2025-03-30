from pipeline.Preprocessing import apply_preprocessing, resize_image
from color_features.feat_extraction import extract_features, extract_col_features, generate_col_dataset
from pipeline.Colourspaces import *
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from pipeline.training import fit_linear_svc, prepare_data

BASE_DIR = "BoWFireDataset"
FIRE_DIR = os.path.join(BASE_DIR, "dataset", "img", "fire")
NONFIRE_DIR = os.path.join(BASE_DIR, "dataset", "img", "not_fire")
MASK_DIR = os.path.join(BASE_DIR, "dataset", "gt", "fire-masks")

def main():
    print("Generating dataset...")
    dataset = generate_col_dataset(FIRE_DIR, NONFIRE_DIR, MASK_DIR, resize_fn=True, blur_fn=True)
    
    print("Preparing data...")
    X_train_scaled, y_train, X_test_scaled, y_test = prepare_data(dataset)
    
    print("Training SVM model with hyperparameter tuning...")
    best_model, best_params = fit_linear_svc(X_train_scaled, y_train)
    
    # print the best parameters in a nice format
    print("\n=== SVM Model Training Summary ===")
    print(f"Training samples: {X_train_scaled.shape[0]}")
    print(f"Feature dimensions: {X_train_scaled.shape[1]}")
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print("\nEvaluating model on test set...")
    y_pred = best_model.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()