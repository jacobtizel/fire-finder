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

    YGreaterThanCb = Y >= Cb
    CrGreaterThanCb = Cr >= Cb

    # meanCalcs = np.logical_and(np.logical_and(Y >= np.mean(Y), Cb <= np.mean(Cb)), Cr >= np.mean(Cr))
    meanCalcs = np.logical_and.reduce([Y >= np.mean(Y), Cb <= np.mean(Cb), Cr >= np.mean(Cr)])

    # CbMinusCrThresh should be between 0 and 224?
    CbMinusCr = (Cb - Cr) >= CbMinusCrThresh 

    chrominanceThresh = np.logical_and(Cb <= 120, Cr >= 150)

    finalMask = np.logical_and.reduce([YGreaterThanCb, CrGreaterThanCb, meanCalcs, CbMinusCr, chrominanceThresh])
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

