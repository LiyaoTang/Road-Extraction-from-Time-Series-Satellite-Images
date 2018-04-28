
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.io
import numpy as np
import h5py
import sys
import gc
import os

from optparse import OptionParser

sys.path.append('../../Visualization/')
from Visualization import *


parser = OptionParser()
parser.add_option("--path", dest="pred_dir")
parser.add_option("--name", dest="pred_name")

parser.add_option("--step", type="int", default=16, dest="step")
parser.add_option("--size", type="int", default=128, dest="size")
parser.add_option("--train", dest="path_train_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_seg_coord_split_;_train")
parser.add_option("--cv", dest="path_cv_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_seg_coord_split_;_cv")
parser.add_option("--test", dest="path_test_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_seg_coord_split_;_test")

parser.add_option("--norm", default='softmax', dest="norm")
parser.add_option("--pred_weight", type="float", default=0.5, dest="pred_weight")
parser.add_option("--analyze_train", action='store_true', default=False, dest="analyze_train")
parser.add_option("--analyze_CV", action='store_true', default=False, dest="analyze_CV")
parser.add_option("--analyze_test", action='store_true', default=False, dest="analyze_test")
parser.add_option("--print_log", action='store_true', default=False, dest="print_log")

parser.add_option("--save", dest="save_path")

(options, args) = parser.parse_args()

pred_dir  = options.pred_dir
pred_name = options.pred_name
step = options.step
size = options.size
path_train_set = options.path_train_set.replace(';', str(size)+'_'+str(step))
path_cv_set    = options.path_cv_set.replace(';', str(size)+'_'+str(step))
path_test_set  = options.path_test_set.replace(';', str(size)+'_'+str(step))

norm        = options.norm
assert(norm in set(['softmax', 'std']))
pred_weight = options.pred_weight
print_log   = options.print_log

analyze_train = options.analyze_train
analyze_CV    = options.analyze_CV
analyze_test  = options.analyze_test

save_path = options.save_path

pred_dir  = pred_dir.strip('/') + '/'
save_path = save_path.strip('/') + '/'
assert not (save_path is None)
if not os.path.exists(save_path): 
    os.makedirs(save_path)

save_name = pred_name.split('.')[0]

h5f = h5py.File(pred_dir + pred_name, 'r')
train_pred = np.array(h5f['train_pred'])
CV_pred = np.array(h5f['CV_pred'])
h5f.close()


# choose a way to normalize
# std
def pred_normalization(pred):
    pred_norm = pred[:,:,1]/pred.sum(axis=-1)
    pred_norm[np.where(pred_norm != pred_norm)] = 0
    pred_norm[np.where(pred_norm == np.float('inf'))] = 1
    return pred_norm

# softmax
def pred_softmax(pred):
    threshold = 500
    pred_exp = pred.copy()
    inf_idx = np.where(pred_exp > threshold)
    
    for x, y in zip(inf_idx[0], inf_idx[1]):
        while((pred_exp[x,y] > threshold).any()):
            pred_exp[x,y] = pred_exp[x,y] / 10
    pred_exp = np.exp(pred_exp[:,:,1])/np.exp(pred_exp).sum(axis=-1)
    pred_exp[np.where(pred[:,:,1] == 0)] = 0 # softmax([0,0]) = (0.5, 0.5)
    return pred_exp

if norm == 'softmax':
    norm_train_pred = pred_softmax(train_pred)
    norm_CV_pred    = pred_softmax(CV_pred)
elif norm == 'std':
    norm_train_pred = pred_normalization(train_pred)
    norm_CV_pred    = pred_normalization(CV_pred)


if analyze_train:
    # Load training set
    train_set = h5py.File(path_train_set, 'r')
    train_raw_image = np.array(train_set['raw_image'])
    train_road_mask = np.array(train_set['road_mask'])
    train_set.close()

    show_pred_prob_with_raw(train_raw_image, norm_train_pred, train_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                            show_plot=False, save_path=save_path + save_name + '_train_' + str(pred_weight).replace('.', '_') + norm + '.png')

    if print_log:
        show_log_pred_with_raw(train_raw_image, norm_train_pred, train_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                            show_plot=False, save_path=save_path + save_name + '_train_' + str(pred_weight).replace('.', '_') + norm + '_log.png')
    plt.close()

if analyze_CV:
    # Load cross-validation set
    CV_set = h5py.File(path_cv_set, 'r')
    CV_raw_image = np.array(CV_set['raw_image'])
    CV_road_mask = np.array(CV_set['road_mask'])
    CV_set.close()
    gc.collect()

    show_pred_prob_with_raw(CV_raw_image, norm_CV_pred, CV_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                            show_plot=False, save_path=save_path + save_name + '_CV_' + str(pred_weight).replace('.', '_') + norm + '.png')

    if print_log:
        show_log_pred_with_raw(CV_raw_image, norm_CV_pred, CV_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                            show_plot=False, save_path=save_path + save_name + '_train_' + str(pred_weight).replace('.', '_') + norm + '_log.png')
    plt.close()


if analyze_test:
    # Load cross-validation set
    test_set = h5py.File(path_test_set, 'r')
    test_raw_image = np.array(test_set['raw_image'])
    test_road_mask = np.array(test_set['road_mask'])
    test_set.close()
    gc.collect()

    show_pred_prob_with_raw(test_raw_image, norm_test_pred, test_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                            show_plot=False, save_path=save_path + save_name + '_CV_' + str(pred_weight).replace('.', '_') + '.png')

    if print_log:
        show_log_pred_with_raw(test_raw_image, norm_test_pred, test_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                            show_plot=False, save_path=save_path + save_name + '_train_' + str(pred_weight).replace('.', '_') + '_log.png')

    plt.close()