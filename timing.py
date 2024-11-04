import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import time  
!wget https://upload.wikimedia.org/wikipedia/commons/f/f9/Zoorashia_elephant.jpg -O elephant.jpg
numTry = 1000
numParam = 25.6e6

with tf.device('/cpu:0'): # if using gpu for preprocessing /gpu:0
  model = ResNet50(weights='imagenet')  # num of params: 25.6 M
  img_path = 'elephant.jpg'
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)

  tic = time.perf_counter()
  for i in range(numTry):
    preds = model.predict(x)

  toc = time.perf_counter()
  totTime=toc-tic
  print("total time:",totTime)
  perRunTime = totTime/numTry
  paramSpeed=numParam/perRunTime 
  # time your preprocessing operation for one sample and multiply with this
  # value to calculate parameter count equivalency 
  print("number of parameters per second:",paramSpeed)