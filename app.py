from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

dic = {0: 'COVID-19', 1: 'NORMAL', 2 :'Viral Pneumonia' }
model = load_model('Xray_v2.h5')

def predict_label(img):
    img = np.array(img)
    img = cv2.resize(img,(224,224))
    # as some images are png so conv to jpg
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img = img.reshape(1,224,224,-1)
    img = preprocess_input(img)
    p = np.argmax(model.predict(img), axis=1)
    return dic[p[0]]
    
@app.route("/", methods=['GET', 'POST'])
def home():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        img = Image.open(request.files['my_image'])
        p = predict_label(img)
    return render_template("index.html", scrollToAnchor = 'scan' ,prediction=p)

if __name__ =='__main__':
    app.run()