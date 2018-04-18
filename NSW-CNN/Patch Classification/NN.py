# coding: utf-8

import numpy as np
import sklearn as sk
import sklearn.linear_model as sklm
import sklearn.metrics as skmt
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.io
import h5py
import sys
import os
import gc
import os
import psutil
import tensorflow as tf

from optparse import OptionParser

sys.path.append('../Metric/')
sys.path.append('../../Visualization/')
sys.path.append('../../Data_Preprocessing/')
from Metric import *
from Visualization import *
from Data_Extractor import *


parser = OptionParser()
parser.add_option("--train", dest="path_train_set")
(options, args) = parser.parse_args()

path_train_set = options.path_train_set

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage before data loaded:', process.memory_info().rss / 1024/1024, 'MB')
print()



''' Data preparation '''




# Load
data = h5py.File(path_train_set, 'r')
X = np.array(data['image_patch'])
Y = np.array(data['road_existence'])
data.close()

# Construct training & test set
index_mask = np.arange(X.shape[0])
np.random.shuffle(index_mask)

train_index = index_mask[:int(index_mask.size*0.75)]
test_index = index_mask[int(index_mask.size*0.75):]

train_x = X[train_index].flatten().reshape((train_index.size, -1))
train_y = Y[train_index]

test_x = X[test_index].flatten().reshape((test_index.size, -1))
test_y = Y[test_index]

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)



width = 28
height = 28
band = 7

L1_out = 512
L2_out = 256
L3_out = 128
L4_out = 64
class_output = 1 # number of possible classifications for the problem

batch_size = 64
learning_rate = 9e-6
iteration = 50000

# Normalize Parameters
mu = train_x.mean(axis=0, keepdims=True)
sigma = 0
for img in train_x:
    sigma += (img-mu)**2
sigma /= train_x.shape[0]



x = tf.placeholder(tf.float32, shape=[None, width*height*band], name='x')
y = tf.placeholder(tf.float32, shape=[None, class_output], name='y')


# Layer 1
net = tf.contrib.layers.fully_connected(inputs=x, num_outputs=L1_out)

# Layer 2
net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=L2_out)

# Layer 3
net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=L3_out)


net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=class_output, activation_fn=None)

net_pred = tf.nn.sigmoid(net)
prediction = tf.cast(tf.round(net_pred), tf.int32, name='prediction')


# Cost function & optimizer:

# In[9]:


cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=net, labels=y)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


# In[10]:


accuracy = tf.reduce_mean(tf.cast(tf.equal(y, tf.round(net_pred)), "float"))


# Train & monitor:


saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_mask = np.arange(train_x.shape[0]) # shuffle the dataset
np.random.shuffle(train_mask)
batch_num = int(train_mask.size/batch_size)

learning_curve = []
for i in range(iteration):
    start = i%batch_num * batch_size
    end = start + batch_size

    if end > train_mask.size:
        end = train_mask.size
        np.random.shuffle(train_mask)
    
    index = train_mask[start:end]    
    batch = [((train_x[index]-mu)/sigma), np.matrix(train_y[index]).astype(int).T]

    # snap shot
    if i%1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1]})
        learning_curve.append(train_accuracy)
        
    train_step.run(feed_dict={x: batch[0], y: batch[1]})
print("finish")



# plot training curve
print(learning_curve)

# Evaluation

train_metric = Metric()

batch_num = int(train_mask.size/batch_size)+1
for i in range(batch_num):
    start = i%batch_num * batch_size
    end = start + batch_size

    if end > train_mask.size:
        end = train_mask.size
    
    batch = [((train_x[start:end]-mu)/sigma), np.matrix(train_y[start:end]).T]
    
    # record metric   
    pred = prediction.eval(feed_dict={x:batch[0], y: batch[1]})
    train_metric.accumulate(pred, batch[1])
    
train_metric.print_info()


# In[ ]:
test_mask = np.arange(test_x.shape[0])
np.random.shuffle(test_mask)
batch_num = int(test_mask.size/batch_size)+1

test_metric = Metric()
test_acc = []
for i in range(batch_num):
    start = i%batch_num * batch_size
    end = start + batch_size

    if end > test_mask.size:
        end = test_mask.size
    
    batch = [((test_x[start:end]-mu)/sigma), np.matrix(test_y[start:end]).T]

    test_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1]})
    test_acc.append(test_accuracy * (end-start))

    # record metric   
    pred = prediction.eval(feed_dict={x:batch[0], y: batch[1]})        
    test_metric.accumulate(pred, batch[1])
    
test_metric.print_info()
print(sum(test_acc)/test_x.shape[0])
