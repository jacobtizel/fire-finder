import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
from scipy import ndimage as ndi

from skimage import data
from skimage import io, color
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.transform import resize
from sklearn.cluster import KMeans


def createGaborFilter(image, shrinkImage: bool):
    startTime = time.time()
    image = img_as_float(image)
    imageBW = color.rgb2gray(image)
    
    #Shrinking the image takes execution from 4.25 -> 0.67 Seconds. Definitely consider using if resolution isn't hugely important
    if shrinkImage:
        shrink = (slice(0, None, 3), slice(0, None, 3))
        imageBW = imageBW[shrink]
    
    
    #0, 45, 90, 135.
    theta1 = [0, np.pi/4, np.pi/2, 3*np.pi/4]# Choose a variety of theta values for good filter kernals
    
    sigma = range(1, 4) # Define a standard deviation? for the kernal. Need to double check this
    frequency = [0.05, 0.10, 0.15, 0.20, 0.25] #Manual Frequency finding. Maybe this depends on the image?
    kernels = createFilterKernels(theta1, sigma, frequency)
    
    gaborFeatures = [] # This will store image features. This can be used for classification

    for kernel in kernels:
        response = gaborPower(imageBW, kernel)
        gaborFeatures.append(response.flatten())
        
    gaborFeatures = np.array(gaborFeatures).T #Reshape to (pixels, feature(s))
    #With features this might be complete.
    #Add segmentation for testing? Maybe this stays?
    segmentedImage = applySegmentation(gaborFeatures,imageBW)
    
    showSegmentedImage(image,segmentedImage,startTime)

def createFilterKernels(thetaVals: list, sigmaVals:list, frequencyVals:list):
# prepare filter bank kernels
    kernels = []
    for theta in thetaVals:
        for sigma in sigmaVals:
            for frequency in frequencyVals:
                kernel = np.real(
                    gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                )
                kernels.append(kernel)
    return kernels

def gaborPower(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(
        ndi.convolve(image, np.real(kernel), mode='wrap') ** 2
        + ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2
    )

def applySegmentation(featureMtx, image):
    clusters = 2
    #Create Kmeans clustering object. Seed to a state for deterministic behaviour
    kmeans = KMeans(n_clusters= clusters, random_state=42, n_init='auto')
    kmeans.fit(featureMtx) # Fit features into clusters
    segmentedImage = kmeans.labels_.reshape(image.shape)
    return segmentedImage

def showSegmentedImage(image, segmentedImage, startTime: int) -> None: 
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Fire Image')
    axes[0].axis('off')

    # Plot segmented image with different regions
    axes[1].imshow(segmentedImage, cmap='viridis')
    axes[1].set_title('Segmented Fire Image')
    axes[1].axis('off')
    
    endTime = time.time()
    print(f"Execution took: {endTime-startTime:.2f} Seconds ")
    plt.tight_layout()
    plt.show()



#This is for testing the Gabor Filter Independently
fireImageLocation = 'BoWFireDataset/dataset/img/fire/fire027.png'
fire = cv.imread(fireImageLocation) #BGR and not RGB?????
fire = cv.cvtColor(fire, cv.COLOR_BGR2RGB)
#Maybe increasing contrast and brightness will help?

#fire = fire[shrink]
createGaborFilter(fire, False)

