import pickle
import tensorflow as tf
import facenet
import pickle
import os
import detect_face

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