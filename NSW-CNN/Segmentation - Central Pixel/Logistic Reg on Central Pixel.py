# encoding: utf-8

import numpy as np
import sklearn as sk
import sklearn.linear_model as sklm
import sklearn.ensemble as sken
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.io
import h5py
import sys
import os

from optparse import OptionParser

sys.path.append('../Metric/')
sys.path.append('../../Visualization/')
sys.path.append('../../Data_Preprocessing//')
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

(options, args) = parser.parse_args()

path_train_set = options.path_train_set
path_cv_set = options.path_cv_set
save_path = options.save_path
model_name = options.model_name
use_weight = options.use_weight
pos_num = options.pos_num
step = options.step

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
if not step:
	step = 8
if not epoch:
	epoch = 15

if not model_name:
	model_name = "sk-SGD_"
	if use_weight: model_name += "weight_"
	model_name += "s" + str(step) + "_"
	model_name += "pos" + str(pos_num) + "_"
	model_name += "ep" + str(epoch)
	
	print("will be saved as ", model_name)
	print("will be saved into ", save_path)


''' Data preparation '''


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
CV_Data = Data_Extractor (CV_raw_image, CV_road_mask, step,
						  pos_topleft_coord = CV_pos_topleft_coord,
						  neg_topleft_coord = CV_neg_topleft_coord)

print("train data:")
print(train_raw_image.shape, train_road_mask.shape)
print("pos = ", Train_Data.pos_size, "neg = ", Train_Data.neg_size)
print("cv data:")
print(CV_raw_image.shape, CV_road_mask.shape)
print("pos = ", CV_Data.pos_size, "neg = ", CV_Data.neg_size)


''' Create model '''


# model parameter
width = step
height = step
band = 7

batch_size = 64
learning_rate = 9e-6
iteration = int(Train_Data.size / batch_size) + 1

# create SGD classifier
if use_weight:
	log_classifier = sklm.SGDClassifier(loss='log', max_iter=1, 
	                                    class_weight={0:Train_Data.pos_size/Train_Data.size,
	                                                  1:Train_Data.neg_size/Train_Data.size})
else:
	log_classifier = sklm.SGDClassifier(loss='log', max_iter=1)

all_classes = np.array([0, 1])
print(log_classifier)
sys.stdout.flush()


''' Train & monitor '''


learning_curve = []
for epoch_num in range(epoch):
    for iter_num in range(iteration):

        batch_x, batch_y = Train_Data.get_patches(batch_size=batch_size, positive_num=pos_num, norm=True)
        batch_x = batch_x.reshape((batch_size, -1))
        
        log_classifier.partial_fit(batch_x, batch_y, all_classes)

    # snap shot on CV set
	cv_metric = Metric()
	for x, y in CV_Data.iterate_data(norm=True):
	    x = x.reshape((1, -1))
	    pred = log_classifier.predict(x)
	    cv_metric.accumulate(np.array([pred]), np.array([y]))
	balanced_acc = cv_metric.get_balanced_acc()
    
    learning_curve.append(balanced_acc)
    print(" balanced_acc = ", balanced_acc)
    sys.stdout.flush()

print("finish")

# plot training curve
plt.figsize=(9,5)
plt.plot(learning_curve)
plt.title('learning_curve_on_cross_validation')
plt.savefig(save_path+'Analysis/'+model_name+'learning_curve.png', bbox_inches='tight')
plt.close()

from sklearn.externals import joblib
joblib.dump(log_classifier, save_path+model_name) 

saved_sk_obj = joblib.load(save_path+model_name)
assert (saved_sk_obj.coef_ == log_classifier.coef_).all()


''' Evaluate model '''


print(log_classifier.coef_.shape)
print(log_classifier.coef_.max(), log_classifier.coef_.min(), log_classifier.coef_.mean())

# train set eva
train_metric = Metric()
for x, y in Train_Data.iterate_data(norm=True):
    x = x.reshape((1, -1))
    pred = log_classifier.predict(x)
    train_metric.accumulate(np.array([pred]), np.array([y]))
    
train_metric.print_info()

# cross validation eva
cv_metric.print_info()
sys.stdout.flush()


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