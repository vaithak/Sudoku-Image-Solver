import cv2
import numpy as np

def basic_preprocessing(img: np.ndarray) -> np.ndarray:
    # create a CLAHE object for Histogram equalisation and improvng the contrast.
    img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(8,8))
    enhanced = clahe.apply(img_plt)

    # Edge preserving smoother:
    # https://dsp.stackexchange.com/questions/60916/what-is-the-bilateral-filter-category-lpf-hpf-bpf-or-bsf
    x, y = max(img.shape[0]//200, 5), max(img.shape[1]//200, 5)
    blurred = cv2.GaussianBlur(enhanced, (x+(x+1)%2, y+(y+1)%2), 0)
    blurred = cv2.bilateralFilter(blurred,7,75,75)
    return blurred

# requires a grayscale image as input
def to_binary(img: np.ndarray) -> np.ndarray:
    # opening for clearing some noise
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)

    thresholded_img = cv2.adaptiveThreshold(opened, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
    inverted = cv2.bitwise_not(thresholded_img)

    if(img.shape[0] > 1000 and img.shape[1] > 1000):
        se = np.ones((2,2))
        eroded = cv2.erode(inverted, se, iterations=1)
    else:
        se = np.ones((2,2))
        eroded = cv2.erode(inverted, se, iterations=1)
    
    return eroded


def processImage(img: np.ndarray) -> np.ndarray:
    preprocessed = basic_preprocessing(img)
    binary = to_binary(preprocessed)

    if(img.shape[0] > 1000 and img.shape[1] > 1000):
      kernel = np.ones((3,3))
      dilated = cv2.dilate(binary, kernel, iterations=3)
      eroded = cv2.erode(dilated, kernel, iterations=3)
    else:
      eroded = binary

    return eroded
