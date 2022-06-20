import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import resnet50
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from skimage.util import view_as_blocks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model=load_model('C:/Users/Mnight/wtsres50.hdf5')

def pred(img):
    img.resize(256,256,3)
    img=/(1./255)
    img=np.expand_dims(img,axis=0)
    a=model.predict(img)
    score=tf.nn.softmax(a)
    c=np.argmax(score)
    return max(score)*100,c

if __name__ == '__main__':
    load_saved_artifacts()

