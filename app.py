from flask import Flask, render_template,url_for, request,redirect,send_file
from werkzeug.utils import secure_filename
import os
from os import path
import base64
import tensorflow
from tensorflow.keras.models import load_model
import cv2
import numpy as np
####################################################################################

def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)






app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():

    return render_template("home.html")



@app.route("/convert", methods=["GET", "POST"] )

def convert():
    if request.method == "POST":

       question = request.form.get("question")

       if question == "":
           return render_template("ez.html") 


       

       f = question[22:]
       img_data = f.encode()
       model = load_model('final_model.h5')


       with open("picture.png", "wb") as fh:
           fh.write(base64.decodebytes(img_data))
       
       m = read_transparent_png('picture.png')

       gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
       gray =  cv2.bitwise_not(gray)
       final = cv2.resize(gray, (28,28) )
       pred = model.predict(final.reshape(1, 28, 28, 1))
       ans = pred.argmax()
       print(pred.argmax())


       return render_template("fin.html", ans=ans, question=question)
    #    return render_template("answerr.html" , ans=ans, text=text , question=question, answer_list=answer_list,scores=scores)

    return render_template("ez.html")   
    # return render_template("convert.html" , text = text1)





@app.route("/about")
def about():
	return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
