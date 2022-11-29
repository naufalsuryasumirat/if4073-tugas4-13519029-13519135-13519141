import keras
import pickle
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

path = Path(__file__).resolve().parent

model: keras.Model = keras.models.load_model(f'{path/"models/baseline_cnn_model_4"}', compile=False)

pred_labels = {}
with open(f'{path/"dict_softmax/baseline_cnn_model_4.pkl"}', 'rb') as fl:
  pred_labels = pickle.load(fl)

def predict_fruit(path: str) -> Tuple[str, List[Tuple[str, float]]]:
  image = cv.imread(path)
  image = cv.resize(image, dsize=(224, 224))
  image = image.astype('float64') / 255
  image = image.reshape(1, 224, 224, 3)
  image = tf.convert_to_tensor(image, dtype='float64')
  image = tf.keras.applications.vgg16.preprocess_input(image)

  pred: tf.Tensor = model.predict(image)
  sureness = [(pred_labels[i], conf) for (i, conf) in enumerate(pred[0])]
  pred = np.argmax(pred, axis=1)
  
  return pred_labels[pred[0]], sureness

if __name__ == '__main__':
  print(predict_fruit('data/test/banana/331.jpg'))
