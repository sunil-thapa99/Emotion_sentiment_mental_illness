from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import randint

import tensorflow as tf
import cv2
import numpy as np
import facenet
import detect_face
import os
import pickle
import json
import time
import datetime
import math
# from PIL import Image
from scipy.ndimage import rotate
from sklearn.svm import SVC
from keras.models import load_model
from keras.preprocessing.image import img_to_array

emotion_model_path = 'models/emotion_XCEPTION.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]


# from gtts import gTTS 



class LoadModel():
    """  Importing and running isolated TF graph """
    def __init__(self):
        # Create local graph and use it in the session
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))

            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, os.getcwd() + '/align')

                self.modeldir = os.getcwd() + '/models/facenet/20170512-110547.pb'
                facenet.load_model(self.modeldir)

                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

                self.classifier_filename = os.getcwd() + '/models/classifier.pkl'
                self.classifier_filename_exp = os.path.expanduser(self.classifier_filename)

                with open(self.classifier_filename_exp, 'rb') as infile:
                    (self.model, self.class_names) = pickle.load(infile)
                    print('load classifier file-> %s' % self.classifier_filename_exp)


    def embedding_tensor(self):
        return(self.embedding_size)

    def nets(self):
        return(self.pnet, self.rnet, self.onet)


    def predict(self, data, emb_array):
        """ Running the activation operation previously imported """

        feed_dict = {self.images_placeholder: data, self.phase_train_placeholder: False}
        emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)
        predictions = self.model.predict_proba(emb_array)

        return(predictions)



model = LoadModel()
minsize = 75  # minimum size of face
threshold = [0.6, 0.8, 0.92]  # three steps's threshold
factor = 0.709  # scale factor
image_size = 182
input_image_size = 160
pnet, rnet, onet = model.nets()
embedding_size = model.embedding_tensor()


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
id = None
lable = "None"
attendance_sheet= dict()
def recognize(img):
    global name, label

    frame = img

    nameList = []
    detect_box = dict()

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
                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (255, 0, 0), 1)
                # print(name)
                id = int(best_class)
                now = datetime.datetime.now()
                checkin_time = now.strftime("%H:%M:%S")

                mytime = checkin_time.split(':',-1)
                mytime = int(mytime[0])
                speech = 'Good Morning: {}'.format(name)
                # if mytime < 15:
                #     name = 'Hello ' + name
                # else:
                #     name = 'Bye Bye ' + name
                    
                accuracy = str(best_class_probabilities)
                # if str(id) not in attendance_sheet.keys():
                #     attendance_sheet.update({str(id): {'Id': str(id), 'Checkin Time': checkin_time}})
                #     attendance_register(id)
                #     sound = gTTS(text=speech, lang='en', slow=False) 
                #     sound.save('audio.mp3')
                #     os.system("mpg321 audio.mp3")
                    # pass
                detect_box.update({str(id): [name, str(id), bb[i][0], bb[i][1]]})
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
                
                # else:
                #     detect_box.get(str(id)).update([name, str(id), bb[i][0], bb[i][1]])

        for i in detect_box.keys():
            # print(detect_box.get(i))
            text = "{}".format(detect_box.get(i)[0])
            cv2.putText(frame, text, (detect_box.get(i)[2]+10, detect_box.get(i)[3]-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))

        
    # except:
    #     pass
    return frame


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    _, frame = cam.read()
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame.shape[1],frame.shape[0]))

    while True:
        ret, frame = cam.read()
        frame = recognize(frame)
        print(attendance_sheet)
        cv2.imshow('Image', frame)
        # out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # out.release()
    cam.release()
    cv2.destroyAllWindows()
