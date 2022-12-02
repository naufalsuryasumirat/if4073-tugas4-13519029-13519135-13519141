import cv2 as cv
import numpy as np
import pandas as pd
from skimage import color
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
from typing import List, Union, Tuple

def imshape(mat: List[List[Union[List[int], int]]]) -> Tuple[int, int, int]:
  shapes = np.shape(mat)
  dim = len(shapes)
  return *shapes[:2], shapes[2] if dim > 2 else None

def load_image(path: str) -> List[List[List[float]]]:
  img = imread(path)
  return resize(img, (128*4, 64*4))

def proc_segment(image) -> None:
  img = (np.array(image) * 255).round().astype(np.uint8)
  edge = cv.Canny(img, threshold1=0, threshold2=255)
  edge = cv.dilate(edge, None)
  contour = sorted(cv.findContours(edge, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2],
                   key=cv.contourArea)[-1]
  mask = cv.drawContours(np.zeros(imshape(edge)[:2], np.uint8), [contour], -1, 255, -1)
  segmented = cv.bitwise_and(img, img, mask=mask)
  return segmented

def gray_image(image: List[List[List[int]]]) -> List[List[List[int]]]:
  return color.rgb2gray(np.array(image)[:,:,:3])

def proc_hog(image) -> None:
  fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16,16),
                      cells_per_block=(2,2), visualize=True,
                      feature_vector=True, block_norm='L2')
  return fd, hog_image


if __name__ == '__main__':
  print()
