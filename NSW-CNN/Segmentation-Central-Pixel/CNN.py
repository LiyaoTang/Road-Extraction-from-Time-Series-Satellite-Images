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

parser.add_option("--train", dest="path_train_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_16_train")
parser.add_option("--cv", dest="path_cv_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_16_cv")

parser.add_option("--pos", type="int", default=0, dest="pos_num")
parser.add_option("--size", type="int", default=16, dest="size")
parser.add_option("-e", "--epoch", type="int", default=15, dest="epoch")
parser.add_option("--learning_rate", type="float", default=6e-9, dest="learning_rate")
parser.add_option("--rand", type="int", default=0, dest="rand_seed")
parser.add_option("--norm", default="", dest="norm")

parser.add_option("--conv", dest="conv_struct")
parser.add_option("--dense", dest="dense_struct")
parser.add_option("--not_weight", action="store_false", default=True, dest="use_weight")
parser.add_option("--use_drop_out", action="store_true", default=False, dest="use_drop_out")
parser.add_option("--use_center_crop", action="store_true", default=False, dest="use_center_crop")
parser.add_option("--use_batch_norm", action="store_true", default=False, dest="use_batch_norm")

parser.add_option("--gpu", dest="gpu")
(options, args) = parser.parse_args()

path_train_set = options.path_train_set 
path_cv_set = options.path_cv_set
save_path = options.save_path
model_name = options.model_name

pos_num = options.pos_num
norm = options.norm
size = options.size
epoch = options.epoch
learning_rate = options.learning_rate
rand_seed = options.rand_seed

conv_struct = options.conv_struct
dense_struct = options.dense_struct

use_weight = options.use_weight
use_drop_out = options.use_drop_out
use_center_crop = options.use_center_crop
use_batch_norm = options.use_batch_norm

gpu = options.gpu

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

if not model_name:
	model_name = "CNN_"
	model_name += conv_struct + "_"
	model_name += dense_struct + "_"
	if use_weight: model_name += "weight_"
	if use_center_crop: model_name += "crop_"
	if use_drop_out: model_name += "drop_"
	if use_batch_norm: model_name += "bn_"
	if norm: model_name += norm + "_"
	model_name += "p" + str(pos_num) + "_"
	model_name += "e" + str(epoch) + "_"
	model_name += "r" + str(rand_seed)
	
	print("will be saved as ", model_name)
	print("will be saved into ", save_path)

if not conv_struct or not dense_struct:
	print("must provide structure for conv & dense layers")
	sys.exit()
else:
	conv_struct = [int(x) for x in conv_struct.split('-')]
	dense_struct = [int(x) for x in dense_struct.split('-')]
	assert len(conv_struct) == 2 and len(dense_struct) == 1

if norm.startswith('G'): norm = 'Gaussian'
else: norm = 'mean'

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

Train_Data = Data_Extractor (train_raw_image, train_road_mask, size,
							 pos_topleft_coord = train_pos_topleft_coord,
							 neg_topleft_coord = train_neg_topleft_coord,
							 normalization = norm)
# run garbage collector
gc.collect()

CV_Data = Data_Extractor (CV_raw_image, CV_road_mask, size,
						  pos_topleft_coord = CV_pos_topleft_coord,
						  neg_topleft_coord = CV_neg_topleft_coord,
						  normalization = norm)
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

last_conv_flatten = conv_struct[-1]

class_output = 1 # number of possible classifications for the problem
class_weight = [Train_Data.pos_size/Train_Data.size, Train_Data.neg_size/Train_Data.size]
print(class_weight, '[neg, pos]')

batch_size = 64
iteration = int(Train_Data.size/batch_size) + 1

# placeholders
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, size, size, band], name='x')
y = tf.placeholder(tf.float32, shape=[None], name='y')

is_training = tf.placeholder(tf.bool, name='is_training') # batch norm

# Convolutional Layer 1
if use_batch_norm:
	net = tf.contrib.layers.conv2d(inputs=x, num_outputs=conv_struct[0], kernel_size=3, 
	                               stride=1, padding='SAME',
	                               normalizer_fn=tf.contrib.layers.batch_norm,
	                               normalizer_params={'scale':True, 'is_training':is_training},
	                               scope='conv1')
else:
	net = tf.contrib.layers.conv2d(inputs=x, num_outputs=conv_struct[0], kernel_size=3, 
	                               stride=1, padding='SAME', scope='conv1')

net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=2, stride=2, padding='VALID', scope='pool1')

# Convolutional Layer 2
if use_batch_norm:
	net = tf.contrib.layers.conv2d(inputs=net, num_outputs=conv_struct[1], kernel_size=3, 
	                               stride=1, padding='SAME',
	                               normalizer_fn=tf.contrib.layers.batch_norm,
	                               normalizer_params={'scale':True, 'is_training':is_training},
	                               scope='conv2')
else:
	net = tf.contrib.layers.conv2d(inputs=net, num_outputs=conv_struct[1], kernel_size=3, 
                               stride=1, padding='SAME', scope='conv2')

net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=2, stride=2, padding='VALID', scope='pool2')

# Flattening
net = tf.contrib.layers.flatten(net, scope='flatten')

# Dense Layer 1
net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=dense_struct[0], scope='dense1')

if use_drop_out:
	net = tf.contrib.layers.dropout(inputs=net, keep_prob=0.5, is_training=is_training, scope='dropout')

# Dense Layer 2 (output)
if use_center_crop:
	center_crop = tf.contrib.layers.flatten(x[:, int(size/2)-1:int(size/2)+2, int(size/2)-1:int(size/2)+2, :])
	net = tf.concat( [net, center_crop], 1)
net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=class_output, activation_fn=tf.nn.sigmoid, scope='dense2')
logits = tf.squeeze(net, name='logits')

# calculate entropy
print(logits)
print(y)
if use_weight:
	cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, 
																			pos_weight=class_weight[1]))
else:
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # collect update_ops into train step
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after model created:', process.memory_info().rss / 1024/1024, 'MB')
print()
sys.stdout.flush()



''' Train & monitor '''



saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

balanced_acc_curve = []
AUC_curve = []
avg_precision_curve = []
cross_entropy_curve = []
for epoch_num in range(epoch):
	for iter_num in range(iteration):

		batch_x, batch_y = Train_Data.get_patches(batch_size=batch_size, positive_num=pos_num, norm=True)
		batch_x = batch_x.transpose((0, 2, 3, 1))

		train_step.run(feed_dict={x: batch_x, y: batch_y, is_training: True})

	# snap shot on CV set
	cv_metric = Metric_Record()
	cv_cross_entropy_list = []
	for batch_x, batch_y in CV_Data.iterate_data(norm=True):
		batch_x = batch_x.transpose((0, 2, 3, 1))

		[pred_prob, cross_entropy_cost] = sess.run([logits, cross_entropy], feed_dict={x: batch_x, y: batch_y, is_training: False})
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
for batch_x, batch_y in CV_Data.iterate_data(norm=True):
	batch_x = batch_x.transpose((0, 2, 3, 1))

	[pred_prob, cross_entropy_cost] = sess.run([logits, cross_entropy], feed_dict={x: batch_x, y: batch_y, is_training: False})
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
train_pred_road = np.zeros(train_road_mask.shape)
for coord, patch in Train_Data.iterate_raw_image_patches_with_coord(norm=True):
	patch = patch.transpose((0, 2, 3, 1))
	train_pred_road[int(coord[0]+size/2), int(coord[1]+size/2)] = logits.eval(feed_dict={x: batch_x, y: batch_y, is_training: False})

# Predict road prob on CV
CV_pred_road = np.zeros(CV_road_mask.shape)
for coord, patch in CV_Data.iterate_raw_image_patches_with_coord(norm=True):
	patch = patch.transpose((0, 2, 3, 1))
	CV_pred_road[int(coord[0]+size/2), int(coord[1]+size/2)] = logits.eval(feed_dict={x: batch_x, y: batch_y, is_training: False})

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