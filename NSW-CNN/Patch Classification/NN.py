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
from Preprocess_Utilities import *

parser = OptionParser()
parser.add_option("--road_type", dest="road_type")
(options, args) = parser.parse_args()

road_type = [int(x) for x in options.road_type.split('-')]

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage before data loaded:', process.memory_info().rss / 1024/1024, 'MB')
print()



''' Data preparation '''




# Load
# data path
route_path = "../Data/090085/"
road_type = np.array(["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "track", # 0-6
                      "residential", "service", "road", "living_street", # 7-10
                      "all_roads"]) # 11 
#                       "motor_trunk_pri_sec_tert_uncl_track", "motor_trunk_pri_sec_tert_uncl"]) # 12-13

path_raw_image = route_path + "090085_20170531.h5"
path_road_mask = np.char.add(np.char.add(np.char.add(route_path+"Road_Data/",road_type),
                                         np.char.add('/', road_type)), '.tif')

# read in raw image
raw_image = np.array(h5py.File(path_raw_image)['scene'])

# read in road mask
road_img_list = []
cnt = 0
for cur_path in path_road_mask:
    print(cnt, cur_path.split('/')[-1])
    cnt += 1
    road_img_list.append(skimage.io.imread(cur_path))

road_img_list = np.array(road_img_list)

# assert 0-1 coding
assert (np.logical_or(road_img_list == 1, road_img_list == 0)).all()

# modify the road mask
print("Used labels:")
combined_road_mask = 0
for i in road_type:
    print(path_road_mask[i].split('/')[-1])
    combined_road_mask += road_img_list[i]
print(combined_road_mask.shape, (combined_road_mask > 1).any())

combined_road_mask[np.where(combined_road_mask > 1)] = 1
assert (np.logical_or(combined_road_mask == 1, combined_road_mask == 0)).all()

image_patch = []
road_patch = []
road_existence = []

for row_offset in [0, 7, 14, 21]:
    for col_offset in [0, 7, 14, 21]:
        cur_img_pch, cur_rd_pch, cur_rd_ex = create_labelled_patches(raw_image, combined_road_mask,
                                                                     row_offset=row_offset,
                                                                     column_offset=col_offset)
        image_patch.extend(cur_img_pch)
        road_patch.extend(cur_rd_pch)
        road_existence.extend(cur_rd_ex)

X = np.array(image_patch)
Y = np.array(road_existence)

print(X.shape, road_patch.shape, Y.shape)

# Construct training & test set
index_mask = np.arange(X.shape[0])
np.random.shuffle(index_mask)

train_index = index_mask[:int(index_mask.size*0.75)]
test_index = index_mask[int(index_mask.size*0.75):]

train_x = X[train_index].flatten().reshape((train_index.size, -1))
train_y = Y[train_index]
train_road_patch = Road_patch[train_index]

test_x = X[test_index].flatten().reshape((test_index.size, -1))
test_y = Y[test_index]
test_road_patch = Road_patch[test_index]

print(train_x.shape, train_y.shape, train_road_patch.shape)
print(test_x.shape, test_y.shape, test_road_patch.shape)

print('class balance: pos=', (road_existence == 1).sum() / road_existence.shape[0], (road_existence == 0).sum() / road_existence.shape[0])
print('in total, ', road_existence.shape[0], ' patches')

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


print(learning_curve)

# Evaluation

train_metric = Metric_Record()
print('On training set')
batch_num = int(train_mask.size/batch_size)+1
for i in range(batch_num):
    start = i%batch_num * batch_size
    end = start + batch_size

    if end > train_mask.size:
        end = train_mask.size
    
    batch = [((train_x[start:end]-mu)/sigma), np.matrix(train_y[start:end]).T]
    
    # record metric   
    pred_prob = net_pred.eval(feed_dict={x:batch[0], y: batch[1]})
    train_metric.accumulate(int(pred_prob>0.5), batch[1], pred_prob)
    
AUC_score = skmt.roc_auc_score(train_metric.y_true, train_metric.pred_prob)
avg_precision_score = skmt.average_precision_score(train_metric.y_true, train_metric.pred_prob)

print('AUC=', AUC_score, 'avg_precision=', avg_precision_score)
train_metric.print_info()


# In[ ]:
print('On test set')
test_mask = np.arange(test_x.shape[0])
np.random.shuffle(test_mask)
batch_num = int(test_mask.size/batch_size)+1

test_metric = Metric_Record()
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
    pred_prob = net_pred.eval(feed_dict={x:batch[0], y: batch[1]})
    train_metric.accumulate(int(pred_prob>0.5), batch[1], pred_prob)

AUC_score = skmt.roc_auc_score(train_metric.y_true, train_metric.pred_prob)
avg_precision_score = skmt.average_precision_score(train_metric.y_true, train_metric.pred_prob)

print('AUC=', AUC_score, 'avg_precision=', avg_precision_score)
test_metric.print_info()