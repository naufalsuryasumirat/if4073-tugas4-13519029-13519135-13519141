import bz2
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from skimage import color
from skimage.io import imread
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.transform import resize
from typing import List, Union, Tuple

path = Path(__file__).resolve().parent

IMAGE = List[List[List[Union[int, float]]]]

file = bz2.BZ2File(f'{path/"models/ml_svm_model_compressed.pbz2"}')
ml_model: SVC = pickle.load(file)
file.close()

def imshape(mat: List[List[Union[List[int], int]]]) -> Tuple[int, int, int]:
  shapes = np.shape(mat)
  dim = len(shapes)
  return *shapes[:2], shapes[2] if dim > 2 else None

def load_image(path: str) -> IMAGE:
  img = imread(path)
  return resize(img, (128*4, 64*4))

def proc_segment(image) -> IMAGE:
  img = (np.array(image) * 255).round().astype(np.uint8)
  edge = cv.Canny(img, threshold1=0, threshold2=255)
  edge = cv.dilate(edge, None)
  contour = sorted(cv.findContours(edge, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2],
                   key=cv.contourArea)[-1]
  mask = cv.drawContours(np.zeros(imshape(edge)[:2], np.uint8), [contour], -1, 255, -1)
  segmented = cv.bitwise_and(img, img, mask=mask)
  return segmented

def gray_image(image: IMAGE) -> IMAGE:
  return color.rgb2gray(np.array(image)[:,:,:3])

def proc_hog(image: IMAGE) -> Tuple[List[float], IMAGE]:
  fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16,16),
                      cells_per_block=(2,2), visualize=True,
                      feature_vector=True, block_norm='L2')
  return fd, hog_image

def proc_color(image: IMAGE) -> Tuple[List[float], Tuple[List[int], List[int], List[int]]]:
  """algorithm to count color values of rgb without counting black colors (segmented)"""
  row, col, dim = imshape(image)
  func = lambda x: x[0] * col + x[1]
  indices = np.where((image[:,:,0] | image[:,:,1] | image[:,:,2]) == 0) # get indices of black pixels
  # using bitwise or of three image channels
  if type(indices) == tuple: indices = np.array(indices).T
  indices = np.apply_along_axis(func, 1, indices)

  red, grn, blu = [np.delete(image[:,:,d], indices) for d in range(dim)]
  hists = [[0 for _ in range(256)] for _ in range(dim)]

  for i, arr in enumerate([red, grn, blu]):
    val, count = np.unique(arr, return_counts=True)
    for j, v in enumerate(val): hists[i][v] = count[j]

  return np.concatenate(hists) / (row * col), hists

def proc_color_2(image: IMAGE) -> Tuple[List[float], Tuple[List[int], List[int], List[int]]]:
  """algorithm 2 to count color values of rgb without counting black colors (segmented)"""
  def getFrequency(img_channel):
    freq = [0 for _ in range(256)]
    for arr in img_channel:
      freq[arr] += 1
    return freq

  temp = []
  for i in range(len(image)):
    for j in range(len(image[0])):
      if image[i][j][0] != 0 or image[i][j][1] != 0 or image[i][j][2] != 0: temp.append(image[i][j])

  temp2 = np.array(temp)
  red = getFrequency(temp2[:,0])
  green = getFrequency(temp2[:,1])
  blue = getFrequency(temp2[:,2])

  return np.concatenate([red, green, blue]) / (len(image) * len(image[0])), (red, green, blue)

def proc_glcm(image: IMAGE) -> IMAGE:
  glcm = graycomatrix((image * 255).round().astype(np.uint8), 
                      distances=[5], 
                      angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                      levels=256,
                      symmetric=True, 
                      normed=True)
  props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM']
  feature = [prop for name in props for prop in graycoprops(glcm, name)[0]]

  return np.array(feature) / 1000 # normalize

def predict_fruit(features: List[float]) -> str:
  return ml_model.predict(features)

if __name__ == '__main__':
  import time

  img = load_image('data/test/apple/Image_2.jpg')
  img = proc_segment(img)

  start = time.time()
  colors_1, _ = proc_color(img)
  print(time.time() - start)

  start = time.time()
  colors_2, _ = proc_color_2(img)
  print(time.time() - start)

  print(all(colors_1 == colors_2))
