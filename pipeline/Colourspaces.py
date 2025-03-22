import cv2
import numpy as np

def YCbCrProcessing(image, CbMinusCrThresh = 80):
    """
    Uses methodologies from "Fire Recognition Using Rgb And YCbCr Color Space"
    to detect potential fire pixels in YCbCr colour space. 
    Returns a mask with the same size as the image.
    """
    YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

    Y, Cr, Cb = cv2.split(YCrCb)

    YgtCb = Y >= Cb
    CrgtCb = Cr >= Cb

    # meanCalcs = np.logical_and(np.logical_and(Y >= np.mean(Y), Cb <= np.mean(Cb)), Cr >= np.mean(Cr))
    meanCalcs = np.logical_and.reduce([Y >= np.mean(Y), Cb <= np.mean(Cb), Cr >= np.mean(Cr)])

    # CbMinusCrThresh should be between 0 and 224?
    CbMinusCr = (Cb - Cr) >= CbMinusCrThresh 

    chrominanceThresh = np.logical_and(Cb <= 120, Cr >= 150)

    finalMask = np.logical_and.reduce([YgtCb, CrgtCb, meanCalcs, CbMinusCr, chrominanceThresh])
    finalMask.dtype="uint8"
    
    return finalMask


def RGBProcessing(image):
    """
    Uses methodologies from "Fire Recognition Using Rgb And YCbCr Color Space"
    to detect potential fire pixels in RGB colour space. 
    Returns a mask with the same size as the image.
    """
    B, G, R = cv2.split(image)

    RgtGgtB = np.logical_and(R > G, G > B)


    meanCalcs = np.logical_and.reduce([R >= np.mean(R), G >= np.mean(G), B <= np.mean(B)])

    finalMask = np.logical_and(RgtGgtB, meanCalcs)
    finalMask.dtype="uint8"
    
    return finalMask

def LABProcessing(image):
    """
    Uses methodologies from "Fire Detection Using Multi Color Space
and Background Modeling" to detect potential fire pixels in CIE L*A*B* colour space.
    Returns a mask with the same size as the image.
    """

    LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    L, A, B = cv2.split(LAB)

    meanCalcs = np.logical_and.reduce([L >= np.mean(L), A <= np.mean(A), B >= np.mean(B)])

    BgtA = B > A

    finalMask = np.logical_and(meanCalcs, BgtA)
    finalMask.dtype="uint8"
    
    return finalMask

def HSVProcessing(image):
    """
    Uses methodologies from "Implementation of Fire Image Processing for Land Fire Detection Using
    Color Filtering Method" to detect fire pixels in HSV colour space.
    Returns a mask with the same size as the image.
    """

    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    H, S, V = cv2.split(HSV)

    H_test = np.logical_and(H >=0, H <= 20)

    S_test = np.logical_and(S >=74, S <= 166)

    V_test = np.logical_and(V >= 200, V <= 230)

    finalMask = np.logical_and.reduce([H_test, S_test, V_test])
    finalMask.dtype="uint8"
    
    return finalMask