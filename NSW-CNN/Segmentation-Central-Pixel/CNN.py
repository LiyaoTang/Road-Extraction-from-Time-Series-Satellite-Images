# encoding: utf-8

import numpy as np
import sklearn as sk
import sklearn.linear_model as sklm
import sklearn.metrics as skmt
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
parser.add_option("--train", dest="path_train_set")
parser.add_option("--cv", dest="path_cv_set")
parser.add_option("--save", dest="save_path")
parser.add_option("--name", dest="model_name")
parser.add_option("--not_weight", action="store_false", dest="use_weight")
parser.add_option("--pos", type="int", dest="pos_num")
parser.add_option("--step", type="int", dest="step")
parser.add_option("-e", "--epoch", type="int", dest="epoch")
parser.add_option("--rand", type="int", dest="rand_seed")
(options, args) = parser.parse_args()

path_train_set = options.path_train_set
path_cv_set = options.path_cv_set
save_path = options.save_path
model_name = options.model_name

use_weight = options.use_weight
pos_num = options.pos_num
step = options.step
epoch = options.epoch
rand_seed = options.rand_seed

if not save_path:
	print("no save path provided")
	sys.exit()
save_path = save_path.strip('/') + '/'
if not os.path.exists(save_path):
	os.makedirs(save_path)
if not os.path.exists(save_path+'Analysis'):
	os.makedirs(save_path+'Analysis')

if not path_train_set:
	path_train_set = "../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_8_train"
if not path_cv_set:
	path_cv_set = "../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_8_cv"

if not pos_num:
	pos_num = 0
if not step:
	step = 8
if not epoch:
	epoch = 15
if not rand_seed:
	rand_seed = 0

if not model_name:
	model_name = "sk-SGD_"
	if use_weight: model_name += "weight_"
	model_name += "s" + str(step) + "_"
	model_name += "p" + str(pos_num) + "_"
	model_name += "e" + str(epoch) + "_"
	model_name += "r" + str(rand_seed)
	
	print("will be saved as ", model_name)
	print("will be saved into ", save_path)

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage before data loaded:', process.memory_info().rss / 1024/1024, 'MB')


''' Data preparation '''


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

Train_Data = Data_Extractor (train_raw_image, train_road_mask, step,
							 pos_topleft_coord = train_pos_topleft_coord,
							 neg_topleft_coord = train_neg_topleft_coord)
# run garbage collector
gc.collect()

CV_Data = Data_Extractor (CV_raw_image, CV_road_mask, step,
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



''' Create model '''



# model parameter
size = step
band = 7

conv_out = [0, 16, 16]
last_conv_flatten = conv_out[-1]
layer_out = [0, 8]

class_output = 1 # number of possible classifications for the problem
class_weight = [Train_Data.pos_size/Train_Data.size, Train_Data.neg_size/Train_Data.size]
print(class_weight, '[neg, pos]')

keep_rate = 0.5 # need regularization => otherwise NaN may appears inside CNN

batch_size = 64
learning_rate = 9e-6
iteration = int(Train_Data.size/batch_size) + 1


# placeholders
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, size, size, band], name='x')
center_crop = tf.contrib.layers.flatten(x[:, int(size/2)-1:int(size/2)+2, int(size/2)-1:int(size/2)+2, :])
y = tf.placeholder(tf.float32, shape=[None], name='y')

keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout
is_training = tf.placeholder(tf.bool, name='is_training') # batch norm

# Convolutional Layer 1
net = tf.contrib.layers.conv2d(inputs=x, num_outputs=conv_out[1], kernel_size=3, 
                               stride=1, padding='SAME',
                               normalizer_fn=tf.contrib.layers.batch_norm,
                               normalizer_params={'scale':True, 'is_training':is_training},
                               scope='conv1')

net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=2, stride=2, padding='VALID', scope='pool1')

# Convolutional Layer 2
net = tf.contrib.layers.conv2d(inputs=net, num_outputs=conv_out[2], kernel_size=3, 
                               stride=1, padding='SAME',
                               normalizer_fn=tf.contrib.layers.batch_norm,
                               normalizer_params={'scale':True, 'is_training':is_training},
                               scope='conv2')

net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=2, stride=2, padding='VALID', scope='pool2')

# Flattening
net = tf.contrib.layers.flatten(net, scope='flatten')

# Dense Layer 1
net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=layer_out[1], scope='dense1')

# # Drop out layer:
# net = tf.contrib.layers.dropout(inputs=net, keep_prob=keep_prob, is_training=is_training, scope='drop')

# Dense Layer 2 (output)
net = tf.concat( [net, center_crop], 1)
net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=class_output, activation_fn=tf.nn.sigmoid, 
                                        scope='dense2')
net = tf.squeeze(net, name='logits')

if use_weight:
	cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=net, targets=y, 
																			pos_weight=class_weight[1]))
else:
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, targets=y))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # collect update_ops into train step
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after model created:', process.memory_info().rss / 1024/1024, 'MB')
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
        
        batch_x, batch_y = Train_Data.get_patches(batch_size=64)
        
        

    print("cross entropy = ", cur_cost)
    learning_curve.append(cur_cost)
print("finish")


#####

for epoch_num in range(epoch):
	for iter_num in range(iteration):

		batch_x, batch_y = Train_Data.get_patches(batch_size=batch_size, positive_num=pos_num, norm=True)
		batch_x = batch_x.transpose((0, 2, 3, 1))

		train_step.run(feed_dict={x: batch_x, y: batch_y, keep_prob: keep_rate, is_training: True})

	# snap shot on CV set
	cv_metric = Metric_Record()
	cv_cross_entropy_list = []
	for x, y in CV_Data.iterate_data(norm=True):
		x = x.reshape((1, -1))

		pred = log_classifier.predict(x)
		pred_prob = log_classifier.predict_proba(x)
		cv_metric.accumulate(Y=y, pred=pred, pred_prob=pred_prob)

		cv_cross_entropy_list.append(cross_entropy.eval(feed_dict={x:batch_x, y: batch_y, keep_prob: 1, is_training:False}))

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

# plot training curve
plt.figsize=(9,5)
plt.plot(balanced_acc_curve, label='balanced_acc')
plt.plot(AUC_curve, label='AUC')
plt.plot(avg_precision_curve, label='avg_precision')
plt.legend()
plt.title('learning_curve_on_cross_validation')
plt.savefig(save_path+'Analysis/'+model_name+'learning_curve.png', bbox_inches='tight')
plt.close()

from sklearn.externals import joblib
joblib.dump(log_classifier, save_path+model_name) 

saved_sk_obj = joblib.load(save_path+model_name)
assert (saved_sk_obj.coef_ == log_classifier.coef_).all()

# run garbage collection
saved_sk_obj = 0
gc.collect()



''' Evaluate model '''


print(log_classifier.coef_.shape)
print(log_classifier.coef_.max(), log_classifier.coef_.min(), log_classifier.coef_.mean())

# train set eva
train_metric = Metric_Record()
for x, y in Train_Data.iterate_data(norm=True):
	x = x.reshape((1, -1))

	pred = log_classifier.predict(x)
	pred_prob = log_classifier.predict_proba(x)
	train_metric.accumulate(Y=y, pred=pred, pred_prob=pred_prob)    
train_metric.print_info()

# cross validation eva
cv_metric.print_info()
sys.stdout.flush()

# run garbage collection
train_metric = 0
cv_metric = 0
gc.collect()

# Predict road mask
index = np.where(log_classifier.classes_ == 1)[0][0]
print(log_classifier.classes_, index)

# Predict road prob masks on train
train_pred_road = np.zeros(train_road_mask.shape)
for coord, patch in Train_Data.iterate_raw_image_patches_with_coord(norm=True):
	patch = patch.reshape([1,-1])
	train_pred_road[int(coord[0]+width/2), int(coord[1]+width/2)] = log_classifier.predict_proba(patch)[0, index]

# Predict road prob on CV
CV_pred_road = np.zeros(CV_road_mask.shape)
for coord, patch in CV_Data.iterate_raw_image_patches_with_coord(norm=True):
	patch = patch.reshape([1,-1])
	CV_pred_road[int(coord[0]+width/2), int(coord[1]+width/2)] = log_classifier.predict_proba(patch)[0, index]

# save prediction
prediction_name = model_name + '_pred.h5'
h5f_file = h5py.File(save_path + prediction_name, 'w')
h5f_file.create_dataset (name='train_pred', data=train_pred_road)
h5f_file.create_dataset (name='CV_pred', data=CV_pred_road)
h5f_file.close()

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after prediction maps calculated:', process.memory_info().rss / 1024/1024, 'MB')

# Analyze pred in plot
show_pred_prob_with_raw(train_raw_image, pred_road, train_road_mask, pred_weight=0.2, figsize=(150,150), 
						show_plot=False, save_path=save_path + 'Analysis/' + model_name + 'prob_road_on_raw - 0_2.png')


# Analyze log pred
log_pred = -np.log(-pred_road + 1 + 1e-7)
print('log pred')
print(log_pred.min(), log_pred.max(), log_pred.mean())

print('normalized log pred')
norm_log_pred = (log_pred - log_pred.min()) / (log_pred.max()-log_pred.min())
print(norm_log_pred.min(), norm_log_pred.max(), norm_log_pred.mean())

show_pred_prob_with_raw(raw_image, norm_log_pred,
						true_road=road_mask, pred_weight=0.2, figsize=(150,150), show_plot=False,
						save_path=save_path + 'Analysis/' + model_name + 'log_prob_on_raw - 0_2.png')
