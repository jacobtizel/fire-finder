fire_dir = "BoWFireDataset/dataset/img"

import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import cv2 as cv

image_path = os.path.join(fire_dir, os.listdir(fire_dir)[0])  
img = cv.imread(image_path)
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# Apply CLAHE to L-channel only
l, a, b = cv.split(lab)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l_eq = clahe.apply(l)
# Merge channels and convert back to BGR
lab_eq = cv.merge([l_eq, a, b])
img_eq = cv.cvtColor(lab_eq, cv.COLOR_LAB2BGR)

plt.figure(figsize=(10, 8))
plt.imshow(img_eq)
plt.axis('off')
plt.title(f'Fire Image Example ({os.path.basename(image_path)})')
plt.show()

print(f"Image shape: {img.shape}")
print(f"Image size (width x height): {img.shape[1]} x {img.shape[0]}")
print(f"Image data type: {img.dtype}")