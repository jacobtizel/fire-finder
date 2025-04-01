import cv2

def closing(mask, ksize=5):
    """
    Applies morphological closing to a binary mask, 
    using a 5x5 elliptical structuring element.
    """
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, strel)
    return closing


def opening(mask, ksize=5):
    """
    Applies morphological opening to a binary mask, 
    using a 5x5 elliptical structuring element.
    """
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, strel)
    return opening

def largestCountourArea(mask):
    """
    Takes a binary mask and returns the area of the largest
    region inside it.
    """
    mask = mask.astype(np.uint8)
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background (label 0)

    if areas.size == 0:
        return 0
    return max(areas)