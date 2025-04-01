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
image= cv2.imread("BoWFireDataset/dataset/img/fire/fire007.png")
#2 resize and bilateral filter (preprocessing step)
img_resized = resize_with_padding(image=image)
blured_img = apply_bilateral_filter(img=img_resized)

height, width, _ = blured_img.shape # to use later after prediction to convert array of pixels to 2D


#3 
blured_img= cv2.cvtColor(blured_img, cv2.COLOR_BGR2RGB) # this step is critical cuz the filters assume input is in RGB not BGR(default)
# 3.1Extract pixel color features : selected colors are ['V', 'b', 'Cr', 'Cb']
color_feats = extract_top_color_features(blured_img)
# 3.2 Extract gabor features : selected gabor resonses are ['g2', 'g6', 'g10', 'g14']
gabor_features= extract_top_gabor_features(blured_img)
combined_feats = np.hstack((color_feats, gabor_features))  # (n_pixels, 8)

#4 need to scale the input: I should have passed the same scaler I used
#  for training or now!! I add this later
scaler = joblib.load("tuned_models/scaler.pkl")
features_scaled = scaler.transform(combined_feats)
#5. Load the saved model and predict
model = joblib.load('tuned_models/best_logistic_model.pkl')
y_pred = model.predict(features_scaled) #1D array of pixels

#6 reshape from array to 2D
mask = y_pred.reshape((height, width))

plt.imshow(mask, cmap='hot')
plt.title("Fire Prediction Mask")
plt.axis('off')
plt.show()

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