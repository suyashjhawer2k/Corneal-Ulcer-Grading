from flask import Flask, request, render_template, url_for,redirect
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)
model=load_model('C:/Users/Mnight/wtsres50.hdf5')

def get_image(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def pred(img):
    np.resize(img,(256,256,3))
    img=img*1./255
    img=np.expand_dims(img,axis=0)
    a=model.predict(img)
    score=tf.nn.softmax(a)
    c=np.argmax(score)
    return max(score)*100,c

@app.route('/')
@app.route('/home',methods=['GET'])
def home():
	return render_template('home.html')

@app.route('/classify',methods=['POST'])
def classify():
	image_data=request.files['image']
	image_b64 = base64.b64encode(image_data.read()).decode('utf-8')
	s='data:image/jpg;base64,'
	image_data=s+image_b64
	img=get_image(image_data)
	cv2.imwrite('static/image.jpg',img)
	sc,c=pred(img)
	score=['Point-Like','Point-Flaky','Flaky']
	return render_template('classify.html',c=sc,score=score[c],eyeimg='static/image.jpg')

@app.route('/contact',methods=['POST'])
def contact():
	return render_template('static/contact.html')

"""@app.errorhandler(404)
def not_found(e):
	return render_template('404.html'),404"""

if __name__ == "__main__":
    app.run(debug=True)
    