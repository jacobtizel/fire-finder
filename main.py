from pipeline.Preprocessing import apply_preprocessing
import pandas as pd
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


DATASET_DIR = "BoWFireDataset/train"

def main():
    # args for preprocessing: (dataset, apply CLAHE, apply DCP, max image size) (experiment with different args later)
    # output: array of fire images and non-fire images as read by opencv in BGR space
    fire_imgs, non_fire_imgs = apply_preprocessing(DATASET_DIR, True, False, None)

    # example of plotting fire image
    example_img = cv.cvtColor(fire_imgs[0], cv.COLOR_BGR2RGB) # we convert BGR to RGB for plotting the image bcs matplotlib works in RGB
    plt.imshow(example_img)
    plt.show()

    # step 2. Convert Images to Different Color Spaces and apply Gabor filters
    print(fire_imgs,non_fire_imgs)


if __name__ == "__main__":
    main()