import cv2
import numpy as np
import operator

# Note: Pass processed images only
def find_largest_contour(img: np.ndarray) -> (bool, np.ndarray):
    # find contours in the edged image, keep only the largest
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grid_cnt = np.array(sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True))
    status, main_contour = False, np.array([])
    if len(grid_cnt) != 0:
        status, main_contour = True, grid_cnt[0]

    return (status, main_contour)


def perspective_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    # Reference: https://stackoverflow.com/questions/57636399/how-to-detect-sudoku-grid-board-in-opencv

    def order_corner_points(corners):
        # Bottom-right point has the largest (x + y) value
        # Top-left has point smallest (x + y) value
        # Bottom-left point has smallest (x - y) value
        # Top-right point has largest (x - y) value
        bottom_r, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in corners]), key=operator.itemgetter(1))
        top_l = (bottom_r + 2)%4
        left_corners = [corners[i] for i in range(len(corners)) if((i!=bottom_r) and (i!=top_l))]
        bottom_l, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in left_corners]), key=operator.itemgetter(1))
        top_r = (bottom_l + 1)%2

        return (corners[top_l][0], left_corners[top_r][0], corners[bottom_r][0], left_corners[bottom_l][0])

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between 
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between 
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in 
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


# Main function for extracting the grid from the image
def extractGrid(processed_img: np.ndarray) -> (bool, np.ndarray):
    status, main_contour = find_largest_contour(processed_img)
    if status == False:
        return (status, main_contour)

    peri = cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, 0.01 * peri, True)
    transformed_processed = perspective_transform(processed_img, approx[0:4])
    # For debugging
    # print(approx)
    # transformed_original = perspective_transform(orig_img, approx)
    # print(approx.shape, approx[0], approx[1], approx[2], approx[3])

    return (True, transformed_processed)
