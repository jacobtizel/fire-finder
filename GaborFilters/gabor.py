import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
import math
from scipy import ndimage as ndi
from PIL import ImageEnhance
from skimage import data
from skimage import io, color
from skimage.util import img_as_float
from skimage.filters import gabor_kernel, gabor, gaussian
from skimage.transform import resize
from sklearn.cluster import KMeans
#from pywt import dwt2



def createGaborFilter(image, shrinkImage: bool) -> np.array:
    """ 
    Creates Gabor Features based on an Input Image.\n
    Uses Gabor Filter kernels (7x7), 1 for every combination of theta, frequency and sigma.
        
    Args:
        image (cv.image): Input image
        shrinkImage (bool): Should the image be shrunk?
        
    Returns:
        np.array: (x, y, n) Where x and y are image dimensions and n is the amount of Gabor Features
    """    
    
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
        response = gaborMagnitudeResponse(imageBW, kernel)
        gaborFeatures.append(response)
        #gaborFeatures.append(response.flatten())
        
    gaborFeatures = np.array(gaborFeatures).T #Reshape to (pixels, feature(s))
    gaborFeatures = np.transpose(gaborFeatures, axes = (1,0,2))
    #gaborFeatures = np.swapaxes(image,1,0)
    stopTime = time.time()
    print(f'Creating Gabor Features took {stopTime-startTime:.2f} Seconds.\n')
    #With features this might be complete.
    #Add segmentation for testing? Maybe this stays?
    # segmentedImage = applySegmentation(gaborFeatures,imageBW)
    # segmentedImage= cv.normalize(segmentedImage, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U) #Convert from int32 to uint8
    # 
    # showSegmentedImage(image,segmentedImage,startTime)
    return gaborFeatures


def createFilterKernels(thetaVals: list, sigmaVals:list, frequencyVals:list) -> list:
    """
    Creates Gabor Filter Kernels.
    Convolve these with a target to apply the filter

    Args:
        thetaVals (list): Range of theta values to use for Filters
        sigmaVals (list): Range of Sigma values to use for Filters
        frequencyVals (list): Range of frequencies/wavelengths to use for Filters

    Returns:
        kernels: List of Gabor Filter Kernals. 
    """    
# prepare filter bank kernels
    kernels = []
    for theta in thetaVals:
        for sigma in sigmaVals:
            for frequency in frequencyVals:
                kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                
                kernels.append(kernel)
    return kernels


def gaborMagnitudeResponse(image, kernel:list):
    """Finds real and imaginary components from filter kernel applied to image then finds magnitude, 
    ||E|| = √(Re(E)^2 + Im(E)^2)
        
    Args:
        image (_MatLike_): Image to apply filter to
        kernel (_List_): List of Gabor Filter kernals to convolve with image

    Returns:
        G_mag: Magnitude of Image response to Gabor Filters
    """    
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(
        ndi.convolve(image, np.real(kernel), mode='wrap') ** 2  
        + ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2
    ) # 

    
            
    
def applySegmentation(featureMtx, image):
    """Applies K-means clustering based on a feature Matrix and an input image

    Args:
        featureMtx (Matrix): List of features
        image (MatLike): Input Image

    Returns:
        MatLike: Binary Image of Clustered Regions
    """    
    clusters = 2
    #Create Kmeans clustering object. Seed to a state for deterministic behaviour
    kmeans = KMeans(n_clusters= clusters, random_state=10, n_init='auto')
    kmeans.fit(featureMtx) # Fit features into clusters
    segmentedImage = kmeans.labels_.reshape(image.shape)
    segmentedImage = cv.bitwise_not(segmentedImage)

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



###------ Ladan's code for gabor features:
#first two functions, uses gray scale and set a kernel size of 7 with different parameters

def get_gabor_kernels(ksize=7, sigma=2.0):
    """Generate a bank of Gabor kernels at different orientations and wavelengths"""
    kernels = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for lambd in [4, 8]:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma=0.5, psi=0)
            kernels.append(kernel)
    return kernels


def apply_gabor_filters(gray_img, kernels):
    """Apply a list of Gabor kernels to a grayscale image, return a list of responses"""
    responses = []
    for kernel in kernels:
        filtered = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
        responses.append(filtered)
    return responses  # List of arrays, shape (H, W) each

## these two functions written to test different kernel sizes and get gabor features for a specific color channels (not grayscale)
def get_diverse_gabor_kernels():
    """Generate a diverse Gabor kernel bank with multiple sizes, wavelengths, orientations, and aspect ratios."""
    kernels = []
    theta_list = np.linspace(0, np.pi, num=4, endpoint=False)
    ksize_list = [11]
    lambd_list = [4, 8]
    gamma_list = [1.0]
    sigma = 1
    psi = 0

    for ksize in ksize_list:
        for theta in theta_list:
            for lambd in lambd_list:
                for gamma in gamma_list:
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
                    kernels.append(kernel)
    return kernels



def apply_gabor_to_selected_channels(img_dict, selected_channels, kernels):
    """
    Apply Gabor filters to selected color channels of an image.

    Args:
        img_dict (dict): Dictionary of image channels, e.g., {'Cb': ..., 'b': ..., 'V': ...}
        selected_channels (list): List of channel names to apply Gabor to
        kernels (list): List of Gabor kernels

    Returns:
        np.ndarray: Flattened vector of all Gabor responses from selected channels
    """
    gabor_feats = []

    for ch in selected_channels:
        gray_img = img_dict[ch]
        for kernel in kernels:
            filtered = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
            gabor_feats.append(filtered.flatten())  # or filtered.mean() for patch-level summary

    return np.concatenate(gabor_feats)



###----------------------------------
#These next lines represent a Matlab like implementation of the Gabor Filter
#This method is very slow, and does not give good results. More work
#could be done to increase the performance of these methods,
# def properGaborMagnitudeResponse(image, theta, frequency, bandwidth, sigma) -> dict:
#     magnitudeDictionary = {}
#     for t in theta:
#         for f in frequency:
#             for s in sigma:
#                 gaborReal, gaborImaginary = gabor(image, frequency=f, bandwidth=bandwidth, theta = t, sigma_x=s, sigma_y=s)
#                 magnitude = getMagnitude([gaborReal,gaborImaginary])
#                 magnitudeDictionary[(t,f)] = magnitude.reshape(image.size)
#     return magnitudeDictionary  
      
# def getMagnitude(response):
#     magnitude = []
#     for i in range(len(response[0])):
#         for j in range(len(response[0][i])):
#             mag = np.sqrt(response[0][i][j]**2+response[1][i][j]**2)
#             magnitude.append(mag)
#     return np.array(magnitude)
    
# def enhanceBrightness(image):
#     brightness = []
#     for x in image:
#             for y in x:
#                 R,G,B = y
#                 bright = np.sqrt(0.5 * R ** 2 + 0.1 * G ** 2 + 0.114 * B ** 2) / 255
#                 brightness.append(bright)
#     avgBrightness = np.mean(brightness)
#     a,b = [0.3, 1]
#     if avgBrightness < 0.1:
#         a = 0.1
#     min_ = 0
#     max_ = 1
#     newBrightness = (b - a) * (avgBrightness- min_) / (max_ - min_) + a
    
#     brightnessFactor = newBrightness/avgBrightness
#     #enhancer = ImageEnhance.Brightness(image)
#     #enhancedImage = enhancer.enhance(brightnessFactor)
#     enhancedImage = cv.convertScaleAbs(image,alpha=brightnessFactor, beta=0)
#     return enhancedImage

# def get_energy_density(pixels):
#     """
#     :param pixels: image array
#     :param size: size of the image
#     :return: Energy density of the image based on its size
#     """
#     energy = get_image_energy(pixels)
#     energy_density = energy / (pixels.shape[0]*pixels.shape[1])
#     return round(energy_density*100,5)
# def get_image_energy(pixels):
#     """
#     :param pixels: image array
#     :return: Energy content of the image
#     """
#     _, (cH, cV, cD) = dwt2(pixels.T, 'db1')
#     energy = (cH ** 2 + cV ** 2 + cD ** 2).sum() / pixels.size
#     return energy

# def createMatlabLikeGaborFilter(image, shrinkImage: bool):
#     startTime = time.time()
#     #enhancedImage = enhanceBrightness(image)
#     # cv.imshow("Enhanced Image",eImage)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()
    
#     #enhancedImage = img_as_float(image)
#     #imageBW = color.rgb2gray(enhancedImage)
#     imageBW = image[:,:,0]
#     imageSize  = [imageBW.shape[0],imageBW.shape[1]]
#     #Shrinking the image takes execution from 4.25 -> 0.67 Seconds. Definitely consider using if resolution isn't hugely important
#     if shrinkImage:
#         shrink = (slice(0, None, 3), slice(0, None, 3))
#         imageBW = imageBW[shrink]
#     #0, 45, 90, 135.
#     thetad = [0, 30, 60, 90, 120, 150]# Choose a variety of theta values for good filter kernals
#     theta1 = []
#     for t in thetad:
#         theta1.append(t*(np.pi/180))   
    
#     sigma = range(1, 4) # Define a standard deviation? for the kernal. Need to double check this
#     #wavelength = [0.05, 0.10, 0.15, 0.20, 0.25] #Manual Frequency finding. Maybe this depends on the image?
#     #From Matlab implementation
#     lambdaMin = 4/math.sqrt(2)
#     lambdaMax = math.sqrt( math.pow(image.shape[0],2) + math.pow(image.shape[1],2)) #Finds Image Hypotenuse
#     step_n = math.floor(math.log2(lambdaMax/lambdaMin))
#     wavelength = []
#     for n in range(0,(step_n-1)):
#         wavelength.append(math.pow(2,n)*lambdaMin)
#     pixels = np.asarray(image, dtype="int32")
#     bandwidth = abs(0.4*get_energy_density(pixels) - 0.5)
#     # kernels = createFilterKernels(theta1, sigma, wavelength)
#     #matKernels = matlabGabor(imageBW,theta1,wavelength,sigma)
#     #matFeats = createGaborFeatureSetMatlab(matKernels,wavelength,sigma)
#     # gaborFeatures = [] # This will store image features. This can be used for classification
#     # for kernel in kernels:
#     #     response = gaborMagnitudeResponse(imageBW, kernel)
#     #     gaborFeatures.append(response.flatten())
    
#     gaborFeatureMagnitudeDictionary = properGaborMagnitudeResponse(imageBW, theta1, wavelength,bandwidth,sigma)
    
#     gaborMagnitude = []
#     for theta, wavelen in gaborFeatureMagnitudeDictionary.items():
#         sigma = 0.5*theta[1]
#         smoothed = wavelen
#         #smoothed = gaussian(wavel, sigma = sigma)
#         gaborMagnitude.append(smoothed)
#     gaborMagnitude = np.array(gaborMagnitude).T
    
#     #gaborFeaturesNew = np.array(gaborMagnitude).reshape((-1, imageSize[0]*imageSize[1])) #Reshape to (pixels, feature(s))
#     #With features this might be complete.
#     #Add segmentation for testing? Maybe this stays?
#     #segmentedImage = applySegmentation(gaborFeatures,imageBW)
#     segmentedImage = applySegmentation(gaborMagnitude,imageBW)
#     #showSegmentedImage(image,segmentedImage,startTime)
#     segmentedImage= cv.normalize(segmentedImage, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U) #Convert from int32 to uint8
#     return segmentedImage

#This is for testing the Gabor Filter Independently
#fireImageLocation = 'BoWFireDataset/dataset/img/fire/fire027.png'
#fire = cv.imread(fireImageLocation) #BGR and not RGB?????
#fire = cv.cvtColor(fire, cv.COLOR_BGR2RGB)
##Maybe increasing contrast and brightness will help?
#
##fire = fire[shrink]
#createGaborFilter(fire, False)
#
