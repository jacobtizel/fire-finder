from pipeline.Preprocessing import apply_preprocessing
from color_features.feat_extraction import extract_features, extract_col_features
from pipeline.Colourspaces import *
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# DATASET_DIR = "BoWFireDataset/train"
DATASET_DIR = "BoWFireDataset/dataset/img/fire"

def main():
    # args for preprocessing: (dataset, apply CLAHE, apply DCP, max image size) (experiment with different args later)
    # output: array of fire images and non-fire images as read by opencv in BGR space
    fire_imgs, non_fire_imgs = apply_preprocessing(DATASET_DIR, False, False, None)



if __name__ == "__main__":
    main()