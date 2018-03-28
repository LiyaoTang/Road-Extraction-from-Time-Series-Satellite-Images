# encoding: utf-8

import numpy as np
import tensorflow as tf
import sklearn.metrics as skmt
import matplotlib
matplotlib.use('agg') # so that plt works in command line
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.io
import h5py
import sys
import os
import gc
import os
import psutil

from optparse import OptionParser

sys.path.append('../Metric/')
sys.path.append('../../Visualization/')
sys.path.append('../../Data_Preprocessing/')
from Metric import *
from Visualization import *
from Data_Extractor import *


parser = OptionParser()
parser.add_option("--save", dest="save_path")
parser.add_option("--name", dest="model_name")

parser.add_option("--train", dest="path_train_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_seg_coord_split_128_train")
parser.add_option("--cv", dest="path_cv_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_seg_coord_split_128_cv")

parser.add_option("--pos", type="int", default=0, dest="pos_num")
parser.add_option("--size", type="int", default=128, dest="size")
parser.add_option("-e", "--epoch", type="int", default=15, dest="epoch")
parser.add_option("--learning_rate", type="float", default=9e-6, dest="learning_rate")
parser.add_option("--batch", type="int", default=2, dest="batch_size")
parser.add_option("--rand", type="int", default=0, dest="rand_seed")

parser.add_option("--conv", dest="conv_struct")
parser.add_option("--not_weight", action="store_false", default=True, dest="use_weight")
parser.add_option("--use_batch_norm", action="store_true", default=False, dest="use_batch_norm")

parser.add_option("--gpu", dest="gpu")
parser.add_option("--gpu_max_mem", type="float", default=0.8, dest="gpu_max_mem")

(options, args) = parser.parse_args()

path_train_set = options.path_train_set
path_cv_set = options.path_cv_set
save_path = options.save_path
model_name = options.model_name

pos_num = options.pos_num
size = options.size
epoch = options.epoch
batch_size = options.batch_size
learning_rate = options.learning_rate
rand_seed = options.rand_seed

conv_struct = options.conv_struct

use_weight = options.use_weight
use_batch_norm = options.use_batch_norm

gpu = options.gpu
gpu_max_mem = options.gpu_max_mem

# restrict to single gpu
assert gpu in set(['0', '1'])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

if not save_path:
	print("no save path provided")
	sys.exit()
save_path = save_path.strip('/') + '/'
if not os.path.exists(save_path):
	os.makedirs(save_path)
if not os.path.exists(save_path+'Analysis'):
	os.makedirs(save_path+'Analysis')

print("Train set:", path_train_set)
print("CV set:", path_cv_set)

if use_batch_norm:
	print("sorry, BN not supported yet")
	sts.exit()

if not model_name:
	model_name = "FCN_incep_"
	model_name += conv_struct + "_"
	if use_weight: model_name += "weight_"
	model_name += "p" + str(pos_num) + "_"
	model_name += "e" + str(epoch) + "_"
	model_name += "r" + str(rand_seed)
	
	print("will be saved as ", model_name)
	print("will be saved into ", save_path)


# parse conv_struct: e.g. 3-16;5-8;1-32 | 3-8;1-16 | ...
# => concat[ 3x3 out_channel=16, 5x5 out_channel=8, 1x1 out_channel=32]
# => followed by inception concat [3x3 out_channel=8, 1x1 out_channel=16]
# => ...
# conv_struct = 1 => use only one 1x1 conv out_channel = classoutput

# note that at last layer, out_channel = 2 is requested
if not conv_struct:
	print("must provide structure for conv")
	sys.exit()
else:
	conv_struct = [[[int(x) for x in config.split('-')] for config in layer.split(';')] for layer in conv_struct.split('|')] 
	assert len(conv_struct) == 3

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage before data loaded:', process.memory_info().rss / 1024/1024, 'MB')
print()



''' Data preparation '''



# set random seed
np.random.seed(rand_seed)

# Load training set
train_set = h5py.File(path_train_set, 'r')
train_pos_topleft_coord = np.array(train_set['positive_example'])
train_neg_topleft_coord = np.array(train_set['negative_example'])
train_raw_image = np.array(train_set['raw_image'])
train_road_mask = np.array(train_set['road_mask'])
train_set.close()

# Load cross-validation set
CV_set = h5py.File(path_cv_set, 'r')
CV_pos_topleft_coord = np.array(CV_set['positive_example'])
CV_neg_topleft_coord = np.array(CV_set['negative_example'])
CV_raw_image = np.array(CV_set['raw_image'])
CV_road_mask = np.array(CV_set['road_mask'])
CV_set.close()

Train_Data = FCN_Data_Extractor (train_raw_image, train_road_mask, size,
							 pos_topleft_coord = train_pos_topleft_coord,
							 neg_topleft_coord = train_neg_topleft_coord)
# run garbage collector
gc.collect()

CV_Data = FCN_Data_Extractor (CV_raw_image, CV_road_mask, size,
						  pos_topleft_coord = CV_pos_topleft_coord,
						  neg_topleft_coord = CV_neg_topleft_coord)
# run garbage collector
gc.collect()

print("train data:")
print(train_raw_image.shape, train_road_mask.shape)
print("pos = ", Train_Data.pos_size, "neg = ", Train_Data.neg_size)
print("cv data:")
print(CV_raw_image.shape, CV_road_mask.shape)
print("pos = ", CV_Data.pos_size, "neg = ", CV_Data.neg_size)

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after data loaded:', process.memory_info().rss / 1024/1024, 'MB')
print()



''' Create model '''



# general model parameter
band = 7

class_output = 2 # number of possible classifications for the problem
if use_weight:
	class_weight = [Train_Data.pos_size/Train_Data.size, Train_Data.neg_size/Train_Data.size]
	print(class_weight, '[neg, pos]')

batch_size = 2
iteration = int(Train_Data.size/batch_size) + 1

tf.reset_default_graph()
with tf.variable_scope('input'):
	x = tf.placeholder(tf.float32, shape=[batch_size, size, size, band], name='x')
	y = tf.placeholder(tf.float32, shape=[batch_size, size, size, class_output], name='y')

	weight = tf.placeholder(tf.float32, shape=[batch_size, size, size], name='class_weight')
	is_training = tf.placeholder(tf.bool, name='is_training') # batch norm


with tf.variable_scope('inception'):
	if conv_struct != 1:
		net = tf.concat([tf.contrib.layers.conv2d(inputs=x, num_outputs=cfg[1], kernel_size=cfg[0], stride=1, padding='SAME') for cfg in conv_struct],
			  					   axis=-1)

net = tf.contrib.layers.conv2d(inputs=input_fuse_map, num_outputs=class_output, kernel_size=1, stride=1, padding='SAME', scope='output_map')
		
with tf.variable_scope('logits'):
	logits = tf.nn.softmax(net)

with tf.variable_scope('cross_entropy')
	logits = tf.reshape(logits, (-1, class_output))
	labels = tf.to_float(tf.reshape(y, (-1, class_output)))

	softmax = tf.nn.softmax(logits) + tf.constant(value=1e-9) # because of the numerical instableness

	if use_weight:
		weight = tf.reshape(weight,[-1])
		cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), weight)) # `*` is element-wise
	else:
		cross_entropy = -tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[1])
	mean_cross_entropy = tf.reduce_mean(cross_entropy, name='mean_cross_entropy')

# Ensures that we execute the update_ops before performing the train_step
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after model created:', process.memory_info().rss / 1024/1024, 'MB')
print()
sys.stdout.flush()



''' Train & monitor '''



saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_max_mem
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

balanced_acc_curve = []
AUC_curve = []
avg_precision_curve = []
cross_entropy_curve = []
for epoch_num in range(epoch):
	for iter_num in range(iteration):

		batch_x, batch_y, batch_w = Train_Data.get_patches(batch_size=batch_size, positive_num=pos_num, norm=True, weighted=True)
		batch_x = batch_x.transpose((0, 2, 3, 1))

		train_step.run(feed_dict={x: batch_x, y: batch_y, weight: batch_w, is_training: True})

	# snap shot on CV set
	cv_metric = Metric_Record()
	cv_cross_entropy_list = []
	for batch_x, batch_y, batch_w in CV_Data.iterate_data(norm=True, weighted=True):
		batch_x = batch_x.transpose((0, 2, 3, 1))

		[pred_prob, cross_entropy_cost] = sess.run([logits, cross_entropy], feed_dict={x: batch_x, y: batch_y, weight: batch_w, is_training: False})
		pred = int(pred_prob > 0.5)

		cv_metric.accumulate(Y=batch_y, pred=pred, pred_prob=pred_prob)
		cv_cross_entropy_list.append(cross_entropy_cost)

	# calculate value
	balanced_acc = cv_metric.get_balanced_acc()
	AUC_score = skmt.roc_auc_score(cv_metric.y_true, cv_metric.pred_prob)
	avg_precision_score = skmt.average_precision_score(cv_metric.y_true, cv_metric.pred_prob)
	mean_cross_entropy = sum(cv_cross_entropy_list)/len(cv_cross_entropy_list)

	balanced_acc_curve.append(balanced_acc)
	AUC_curve.append(AUC_score)
	avg_precision_curve.append(avg_precision_score)
	cross_entropy_curve.append(mean_cross_entropy)

	print("mean_cross_entropy = ", mean_cross_entropy, "balanced_acc = ", balanced_acc, "AUC = ", AUC_score, "avg_precision = ", avg_precision_score)
	sys.stdout.flush()
print("finish")

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after model trained:', process.memory_info().rss / 1024/1024, 'MB')
print()

# plot training curve
plt.figsize=(9,5)
plt.plot(balanced_acc_curve, label='balanced_acc')
plt.plot(AUC_curve, label='AUC')
plt.plot(avg_precision_curve, label='avg_precision')
plt.legend()
plt.title('learning_curve_on_cross_validation')
plt.savefig(save_path+'Analysis/'+'cv_metrics_curve.png', bbox_inches='tight')
plt.close()

plt.figsize=(9,5)
plt.plot(cross_entropy_curve)
plt.savefig(save_path+'Analysis/'+'cv_learning_curve.png', bbox_inches='tight')
plt.close()

# save model
saver.save(sess, save_path + model_name)

# run garbage collection
saved_sk_obj = 0
gc.collect()



''' Evaluate model '''



# train set eva
print("On training set: ")
train_metric = Metric_Record()
train_cross_entropy_list = []
for batch_x, batch_y, batch_w in CV_Data.iterate_data(norm=True, weighted=True):
	batch_x = batch_x.transpose((0, 2, 3, 1))

	[pred_prob, cross_entropy_cost] = sess.run([logits, cross_entropy], feed_dict={x: batch_x, y: batch_y, weight: batch_w, is_training: False})
	pred = int(pred_prob > 0.5)
	
	train_metric.accumulate(Y=batch_y, pred=pred, pred_prob=pred_prob)    
	train_cross_entropy_list.append(cross_entropy_cost)

train_metric.print_info()
AUC_score = skmt.roc_auc_score(train_metric.y_true, train_metric.pred_prob)
avg_precision_score = skmt.average_precision_score(train_metric.y_true, train_metric.pred_prob)
mean_cross_entropy = sum(train_cross_entropy_list)/len(train_cross_entropy_list)
print("mean_cross_entropy = ", mean_cross_entropy, "balanced_acc = ", balanced_acc, "AUC = ", AUC_score, "avg_precision = ", avg_precision_score)

# plot ROC curve
fpr, tpr, thr = skmt.roc_curve(train_metric.y_true, train_metric.pred_prob)
plt.plot(fpr, tpr)
plt.savefig(save_path+'Analysis/'+'train_ROC_curve.png', bbox_inches='tight')
plt.close()

# cross validation eva
print("On CV set:")
cv_metric.print_info()

# plot ROC curve
fpr, tpr, thr = skmt.roc_curve(cv_metric.y_true, cv_metric.pred_prob)
plt.plot(fpr, tpr)
plt.savefig(save_path+'Analysis/'+'cv_ROC_curve.png', bbox_inches='tight')
plt.close()
sys.stdout.flush()

# run garbage collection
train_metric = 0
cv_metric = 0
gc.collect()

# Predict road mask
# Predict road prob masks on train
train_pred_road = np.zeros([x for x in train_road_mask.shape] + [2])
for coord, patch in Train_Data.iterate_raw_image_patches_with_coord(norm=True):
	patch = patch.transpose((0, 2, 3, 1))
	train_pred_road[coord[0]:coord[0]+size, coord[1]:coord[1]+size, :] += logits.eval(feed_dict={x: batch_x, y: batch_y, is_training: False})

# Predict road prob on CV
CV_pred_road = np.zeros([x for x in CV_road_mask.shape] + [2])
for coord, patch in CV_Data.iterate_raw_image_patches_with_coord(norm=True):
	patch = patch.transpose((0, 2, 3, 1))
	CV_pred_road[coord[0]:coord[0]+size, coord[1]:coord[1]+size, :] += logits.eval(feed_dict={x: batch_x, y: batch_y, is_training: False})

# save prediction
prediction_name = model_name + '_pred.h5'
h5f_file = h5py.File(save_path + prediction_name, 'w')
h5f_file.create_dataset (name='train_pred', data=train_pred_road)
h5f_file.create_dataset (name='CV_pred', data=CV_pred_road)
h5f_file.close()

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after prediction maps calculated:', process.memory_info().rss / 1024/1024, 'MB')
print()