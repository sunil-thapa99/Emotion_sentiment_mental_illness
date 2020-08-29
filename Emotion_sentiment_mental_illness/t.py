#Python libraries that we need to import for our bot
import random
from flask import Flask, request
import base64
from flask import jsonify
import base64
import cv2
import numpy as np
import json
from keras.preprocessing.image import img_to_array

import facenet
import detect_face
from LoadModel import LoadModel
from keras.models import load_model

model = LoadModel()
minsize = 75  # minimum size of face
threshold = [0.6, 0.8, 0.92]  # three steps's threshold
factor = 0.709  # scale factor
image_size = 182
input_image_size = 160
pnet, rnet, onet = model.nets()
embedding_size = model.embedding_tensor()

emotion_model_path = 'models/emotion_XCEPTION.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]

with open('train.txt') as json_file:
    HumanNames = json.load(json_file)

l = list(HumanNames)
l.sort()
print(l)

# Crop Padding
left = 1
right = 1
top = 1
bottom = 1
name = "Unknown"
label = "None"
id = None
attendance_sheet= dict()

def recognize(img):
    global name, name

    frame = img

    nameList = []

    if frame.ndim == 2:
        frame = facenet.to_rgb(frame)
    frame = frame[:, :, 0:3]
    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    print('Face Detected: %d' % nrof_faces)
    
    # try:
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        cropped = []
        scaled = []
        scaled_reshape = []
        bb = np.zeros((nrof_faces, 4), dtype=np.int32)

        for i in range(nrof_faces):
            emb_array = np.zeros((1, embedding_size))

            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]

            # inner exceptname = "Unknown"
            #     id = Noneion
            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                continue

            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
            cropped[i] = facenet.flip(cropped[i], False)
            # print(cropped)
            img_resized = cv2.resize(cropped[i], (image_size, image_size))
            # misc.imresize(cropped[i], (image_size, image_size), interp='bilinear')
            scaled.append(img_resized)
            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                    interpolation=cv2.INTER_CUBIC)
            scaled[i] = facenet.prewhiten(scaled[i])
            scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))

            #Call function inside the loaded model
            predictions = model.predict(scaled_reshape[i], emb_array)

            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            # print(best_class_indices)

            try:
                best_class = l[best_class_indices[0]]
            except:
                pass
                # print(best_class_indices[0])
            result_names = HumanNames[best_class]

            # print("ID: ", best_class)
            # print(best_class_probabilities)

            if best_class_probabilities >= 0.85:
                name = result_names

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            print(label, name)
    return jsonify(name=name, label=label)



def readb64(base64_string):
    encoded_data = base64_string.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

app = Flask(__name__)
#We will receive messages that Facebook sends our bot at this endpoint 
@app.route("/index", methods=['GET', 'POST'])
def receive_message():
    img = readb64(request.args.get('url'))
    print(recognize(img))
    return jsonify('Thanks')


if __name__ == "__main__":
    app.run(debug=True)