import os
import keras
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator

path = Path(__file__).resolve().parent

models_path = [
  'baseline_cnn_model_xception_5',
  'baseline_cnn_model_4',
]

model_name = None
for p in models_path:
  if os.path.exists(f'{path/f"models/{p}"}'): model_name = p; break

model: keras.Model = keras.models.load_model(f'{path/f"models/{model_name}"}', compile=False)

datagen = ImageDataGenerator(rescale=1./255)\
  if 'xception' in model_name else\
    ImageDataGenerator(rescale=1./255,
                       preprocessing_function=tf.keras.applications.vgg16.preprocess_input)

pred_labels = {}
with open(f'{path/f"dict_softmax/{model_name}.pkl"}', 'rb') as fl:
  pred_labels = pickle.load(fl)

def predict_fruit(path: str) -> Tuple[str, List[Tuple[str, float]]]:
  image = datagen.flow_from_dataframe(
    dataframe=pd.DataFrame(columns=['Path', 'Label'], data=[[path, '']]),
    x_col='Path',
    y_col='Label',
    target_size=(224,224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=1,
    shuffle=False)

  pred: tf.Tensor = model.predict(image)
  sureness = [(pred_labels[i], conf) for (i, conf) in enumerate(pred[0])]
  pred = np.argmax(pred, axis=1)
  
  return pred_labels[pred[0]], sureness

if __name__ == '__main__':
  print(pred_labels)
  print(predict_fruit('data/test/durian/950.jpg'))
  print(predict_fruit('data/test/lime/197.jpg'))
