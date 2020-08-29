# importing libraries and dataset.
from flask import Flask,render_template,request
# from chatbot.configuration import DATAFILE
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random
from keras.models import load_model
app = Flask(__name__)

from test_model import ModelTester

DATAFILE = r"C:\Users\zerad\Desktop\sujan\git_repo\Retrievalbasedchatbot\chatbot\data\angry.json"
modelpath = r"C:\Users\zerad\Desktop\sujan\git_repo\Retrievalbasedchatbot\chatbot\development\angry_model.h5"
wordspath = r"C:\Users\zerad\Desktop\sujan\git_repo\Retrievalbasedchatbot\chatbot\development\angry_words.pkl"
classpath = r"C:\Users\zerad\Desktop\sujan\git_repo\Retrievalbasedchatbot\chatbot\development\angry_classes.pkl"



@app.route('/')
def home():
    """
    :return: home page
    """
    inp = "angry"
    # inp = request.args.get('msg')
    print("....................", inp, type(inp))
    testobj = ModelTester(DATAFILE, modelpath, wordspath, classpath, inp)
    result = testobj.chat()


    return render_template("home.html",val={'input': inp, 'output':str(result)})

@app.route("/get")
def get_box_response():
    """
    This function is used to get message from the frontend and send back responses to it.
    :return: string of responses.
    """
    # inp = "angry"
    inp = request.args.get('msg')
    print("....................", inp, type(inp))
    testobj = ModelTester(DATAFILE, modelpath, wordspath, classpath, inp)
    result = testobj.chat()
    return str(result)





if __name__ == '__main__':
    """
    This is the part where the program is executed. 
    
    """
    app.run(host = '127.0.0.1', port = 8005,debug=True)
