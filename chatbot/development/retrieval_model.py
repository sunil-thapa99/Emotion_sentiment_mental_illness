#importing libraries
# from chatbot.configuration import DATAFILE
import random
import json
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
lemmatizer = WordNetLemmatizer()
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
import random

DATAFILE = r"C:\Users\zerad\Desktop\sujan\git_repo\Retrievalbasedchatbot\chatbot\data\happy.json"
#creating list
words = list()
classes = list()
documents = list()
ignore_words = ["?","!"]
data = open(DATAFILE,encoding='utf-8').read()
# print(data)
intents = json.loads(data)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print(words)
# print(classes)
# print(documents)

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
print(len(documents),"documents")
print(len(documents),"classes",classes)
print(len(words),"unique lemmatized words",words)
pickle.dump(words,open("happy_words.pkl","wb"))
pickle.dump(classes,open("happy_classes.pkl","wb"))
print("Congratulation!!! Successfully created dumbed data...")

#creating the training data
training = list()
output_empty = [0] * len(classes)
for document in documents:
    bag = list()
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] =1
    training.append([bag,output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("train_x",len(train_x),train_x)
print("train_y",len(train_y),train_y)
print("successfully created training data.")

#creating the model
model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

try:
    model = load_model('happy_model.h5')
    print("Successfully load model . . .")
except:
    sgd = SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)
    model.compile(loss="categorical_crossentropy",optimizer = sgd,metrics=['accuracy'])
    model.fit(np.array(train_x),np.array(train_y),epochs=500,batch_size=8,verbose=1)
    model.save('happy_model.h5')
    print('Successfully created model!!!')

model.summary()
