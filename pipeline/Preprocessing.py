import os
import cv2 as cv
from dcp_dehaze import dcp_dehaze
import cv2


####---Ladan Preprocessing functions
def resize_image(img, size=(256, 256)):
    return cv2.resize(img, size)


def resize_with_padding(image, target_size=(224, 224), pad_color=(0, 0, 0)):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute scale factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize while keeping aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Compute padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad the image to reach target size
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return padded
## gaussian or bilateral? very similar bilateral preserve edges
def apply_gaussian_blur(img, kernel_size=(3, 3), sigma=0.5):
    return cv2.GaussianBlur(img, kernel_size, sigma)

def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def resize(img, max_size: int):
    """
    Resizes an image, preserving the aspect ratio
    """
    M = max(img.shape)
    ratio = float(max_size) / float(M)
    if M > max_size:
        img = cv.resize(img, (0, 0), fx=ratio, fy=ratio)
    return img

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) an img
    
    How: converts to BGR to LAB to seperate the luminance channel, then apply CLAHE to only the luminance channel, then merge back to BGR
    """

    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    
    l, a, b = cv.split(lab)
    
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    
    lab_eq = cv.merge([l_eq, a, b])
    result = cv.cvtColor(lab_eq, cv.COLOR_LAB2BGR)
    
    return result

def apply_DCP_dehaze(image):
    """
    Applies Dark Channel Prior Dehazing to an image
    """
    return dcp_dehaze(image)

def apply_preprocessing(dataset_dir, apply_clahe_flag=True, apply_dcp_flag=False, max_size=None):
    """
    Runs preprocessing to images in dataset_dir, splits them into fire / non-fire categories
    """
    fire_images = []
    non_fire_images = []
    
    files = os.listdir(dataset_dir)
    print(f"Preprocessing {len(files)} images in {dataset_dir}")
    
    for f in files:
        image_path = os.path.join(dataset_dir, f)
        img = cv.imread(image_path)
        
        if img is None:
            print(f"Failed to read img?? this shouldnt happen {image_path}")
            continue
        
        # resize if they wanted it
        if max_size is not None:
            img = resize(img, max_size)
        
        # CLAHE if they wanted it
        if apply_clahe_flag:
            img = apply_clahe(img)
        
        # DCP is they wanted it
        if apply_dcp_flag:
            img = apply_DCP_dehaze(img)
        
        if f.startswith("fire"):
            fire_images.append(img)
        else:
            non_fire_images.append(img)
    
    print(f"Processed {len(fire_images)} fire images and {len(non_fire_images)} non-fire images")
    return fire_images, non_fire_images