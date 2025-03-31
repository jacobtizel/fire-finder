from pipeline.Preprocessing import *
from pipeline.featureExtraction import *
from pipeline.Colourspaces import *
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib


## the following step to test an image on a pre-tained model

#1. Read 
image_bgr = cv2.imread("image_path")
img= cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # this step is critical cuz the filters assume input is in RGB not BGR(default)
#2 resize and bilateral filter (preprocessing step)
img_resized = resize_image(img=img)
blured_img = apply_bilateral_filter(img=img_resized)

height, width, _ = blured_img.shape # to use later after prediction to convert array of pixels to 2D

#3 Extract pixel color features : selected colors are ['V', 'b', 'Cr', 'Cb']


# Load the saved model
#loaded_model = joblib.load('tuned_model/best_logistic_model.pkl')

# Use it to predict
 #img must be resize and preprocessed and specific color and gabor feature extracted.
#color_top = ['Cb', 'Cr', 'b', 'V'] #color to extract
#gabor_top = ['gabor_14', 'gabor_6', 'gabor_10', 'gabor_2'] #gabor responses to extract
#y_pred = loaded_model.predict(img)

# DATASET_DIR = "BoWFireDataset/train"
DATASET_DIR = "BoWFireDataset/dataset/img/fire"

def main():
    # args for preprocessing: (dataset, apply CLAHE, apply DCP, max image size) (experiment with different args later)
    # output: array of fire images and non-fire images as read by opencv in BGR space
    fire_imgs, non_fire_imgs = apply_preprocessing(DATASET_DIR, False, False, None)

    # example of plotting fire image
    # example_img = cv2.cvtColor(fire_imgs[0], cv2.COLOR_BGR2RGB) # we convert BGR to RGB for plotting the image bcs matplotlib works in RGB
    # plt.imshow(example_img)
    # plt.show()

    # step 2. Convert Images to Different Color Spaces and apply Gabor filters
    print(fire_imgs,non_fire_imgs)


    image = fire_imgs[10]

    cv2.imshow('image_window', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ycbcrMask = YCbCrProcessing(image)
    cv2.imshow('image_window', cv2.bitwise_and(image, image, mask=ycbcrMask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rgbMask = RGBProcessing(image)
    cv2.imshow('image_window', cv2.bitwise_and(image, image, mask=rgbMask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    LABMask = LABProcessing(image)
    cv2.imshow('image_window', cv2.bitwise_and(image, image, mask=LABMask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    HSVMask = HSVProcessing(image)
    cv2.imshow('image_window', cv2.bitwise_and(image, image, mask=HSVMask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()