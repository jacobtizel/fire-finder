import cv2

def closing(mask):
    """
    Applies morphological closing to a binary mask, 
    using a 5x5 elliptical structuring element.
    """
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, strel)
    return closing


def opening(mask):
    """
    Applies morphological opening to a binary mask, 
    using a 5x5 elliptical structuring element.
    """
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, strel)
    return opening

def largestCountourArea(mask):
    """
    Takes a binary mask and returns the area of the largest
    region inside it.
    """
    retval, labels, stats, centroids= cv2.connectedComponentsWithStats(mask)
    areas = stats[:, cv2.CC_STAT_AREA]
    
    print(max(areas))
    
    return max(areas)
