# coding: utf-8

import tensorflow as tf
import sklearn as sk
import numpy as np
from sklearn.externals import joblib

class Classifier ():
    def __init__(self, path, name, classifier_type, gpu_max_mem=0.9):
        assert classifier_type in set(['LR', 'FCN'])
        if not path.endswith('/'): path = path + '/'

        self.classifier_type = classifier_type
        if classifier_type == 'LR':
            self.classifier = joblib.load(path+name)
            self.pos_idx = int(np.where(self.classifier.classes_ == 1)[0])
            self.predict = lambda patch: self.classifier.predict_proba(patch.reshape((1, -1)))[0,self.pos_idx]

        else: # FCN
            tf.reset_default_graph()
            tf.train.import_meta_graph(path+name+'.meta')

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = gpu_max_mem
            config.gpu_options.allow_growth = True
            sess = tf.InteractiveSession(config=config)

            saver = tf.train.Saver()
            saver.restore(sess, path+name)
            
            graph = tf.get_default_graph()
            self.x = graph.get_tensor_by_name("input/x:0")
            self.is_training = graph.get_tensor_by_name("input/is_training:0")
            self.prob_out = graph.get_tensor_by_name("prob_out/prob_out:0")
            
            self.predict = lambda patch: self.prob_out.eval(feed_dict={self.x: patch.transpose((0, 2, 3, 1)), 
                                                                       self.is_training: False})[0,:,:]
