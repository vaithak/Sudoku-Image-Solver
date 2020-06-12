import cv2
import numpy as np


def get_square_centers(transformed_img):
  lines_X = np.linspace(0, transformed_img.shape[1], num=10, dtype=int)
  lines_Y = np.linspace(0, transformed_img.shape[0], num=10, dtype=int)
  centers_X = [(lines_X[i] + lines_X[i-1])//2 for i in range(1, len(lines_X))]
  centers_Y = [(lines_Y[i] + lines_Y[i-1])//2 for i in range(1, len(lines_Y))]

  return centers_X, centers_Y

def centering_se(shape: (int, int), shape_ones: (int, int)):
  x = np.zeros(shape)
  assert (shape_ones[0] < shape[0]) and (shape_ones[1] < shape[1])
  width = shape_ones[0]
  height = shape_ones[1]
  
  rows, cols = shape
  for i in range(width):
    for j in range(height):
      x[i][j], x[rows-1-i][j], x[i][cols-1-j], x[rows-1-i][cols-1-j] = 1, 1, 1, 1

  return x


def recentre(img: np.ndarray, prev_center: (int, int), h_se: np.ndarray, v_se: np.ndarray, h_mov_range: (int, int), v_mov_range: (int, int)) -> (int, int):
  # reference: https://web.stanford.edu/class/ee368/Project_Spring_1415/Reports/Wang.pdf
  max_res, max_center = 0, prev_center

  for i in range(v_mov_range[0], v_mov_range[1]):
    curr_center = (prev_center[0] + 0, prev_center[1] + i)
    start_row = max(curr_center[1] - v_se.shape[0]//2, 0)
    start_col = max(curr_center[0] - v_se.shape[1]//2, 0)
    partial = img[start_row:start_row+v_se.shape[0], start_col:start_col+v_se.shape[1]]

    curr_dot = np.sum(partial*(v_se[0:partial.shape[0], 0:partial.shape[1]]))
    # curr_dot = np.sum(img[x1:x1+v_se.shape[0], y1:y1+v_se.shape[1]]*(v_se))
    # print(curr_center, curr_dot)
    if max_res <= curr_dot:
      max_res = curr_dot
      max_center = curr_center

  # # print("max_center after v_se: ", max_center)
  prev_center = max_center
  max_res = 0
  for i in range(h_mov_range[0], h_mov_range[1]):
    curr_center = (prev_center[0] + i, prev_center[1] + 0)
    start_row = max(curr_center[1] - h_se.shape[0]//2, 0)
    start_col = max(curr_center[0] - h_se.shape[1]//2, 0)
    partial = img[start_row:start_row+h_se.shape[0], start_col:start_col+h_se.shape[1]]

    curr_dot = np.sum(partial*(h_se[0:partial.shape[0], 0:partial.shape[1]]))
    # print(curr_center, curr_dot)
    if max_res <= curr_dot:
      max_res = curr_dot
      max_center = curr_center

  # print("max_center after h_se: ", max_center)
  return max_center


def preprocess_digit(digit_img):
  # remove possible edges from border
  digit_img[0:4,:] = 0
  digit_img[:,0:4] = 0
  digit_img[-4:,:] = 0
  digit_img[:,-4:] = 0

  # dilating and eroding the digit
  se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  closed = cv2.morphologyEx(digit_img, cv2.MORPH_CLOSE, se)
  if(np.sum(digit_img) < 255*30):
    closed = np.zeros(closed.shape)

  return closed


def extractDigits(transformed_img):
  centers_X, centers_Y = get_square_centers(transformed_img)
  centers = [(centers_X[i], centers_Y[j]) for i in range(len(centers_X)) for j in range(len(centers_Y))] 
  kernel_shape = (centers_X[1] - centers_X[0], centers_Y[1] - centers_Y[0])
  
  ones_length = (kernel_shape[0]+kernel_shape[1])//20
  v_se = centering_se(kernel_shape, (2,ones_length))
  h_se = centering_se(kernel_shape, (ones_length,2))
  new_centers = []
  for i in range(len(centers)):
    v_mov_range, h_mov_range = (-kernel_shape[0]//8, kernel_shape[0]//8), (-kernel_shape[1]//8, kernel_shape[1]//8)
    if (i<9)            : h_mov_range = (-kernel_shape[1]//32, kernel_shape[1]//8)
    elif (i>71)         : h_mov_range = (-kernel_shape[1]//8, kernel_shape[1]//32)
    if (i%9 == 0)       : v_mov_range = (-kernel_shape[0]//32, kernel_shape[0]//8)
    elif ((i+1)%9 == 0) : v_mov_range = (-kernel_shape[0]//8, kernel_shape[0]//32)
    new_centers.append(recentre(transformed_img, centers[i], h_se, v_se, h_mov_range, v_mov_range))

  digits = []
  for center in new_centers:
    top_l = [center[0]-kernel_shape[1]//2, center[1]-kernel_shape[0]//2]
    top_r = [center[0]+kernel_shape[1]//2, center[1]-kernel_shape[0]//2]
    bottom_l = [center[0]-kernel_shape[1]//2, center[1]+kernel_shape[0]//2]
    bottom_r = [center[0]+kernel_shape[1]//2, center[1]+kernel_shape[0]//2]

    M = cv2.getPerspectiveTransform(np.float32([top_l, top_r, bottom_l, bottom_r]), np.float32([[0,0], [28,0], [0,28], [28,28]]))
    dst = cv2.warpPerspective(transformed_img,M,(28,28))
    dst_mod = preprocess_digit(dst)
    digits.append(dst_mod)

  return digits
