# coding: utf-8

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
sys.path.append('../../Data_Preprocessing/')
from Visualization import *
from Data_Extractor import *

parser = OptionParser()
parser.add_option("--path", dest="pred_dir")
parser.add_option("--name", dest="pred_name")
parser.add_option("--save", dest="save_path")

parser.add_option("--raw", dest="path_raw", default="../../Data/090085/090085_20170531.h5")
parser.add_option("--train", dest="path_train_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_8_train")
parser.add_option("--cv", dest="path_cv_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_8_cv")
parser.add_option("--test", dest="path_test_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_8_test")
parser.add_option("--road", dest="path_road", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/road_mask.tif")

parser.add_option("--analyze_all", action='store_true', default=False, dest="analyze_all")
parser.add_option("--analyze_train", action='store_true', default=False, dest="analyze_train")
parser.add_option("--analyze_CV", action='store_true', default=False, dest="analyze_CV")
parser.add_option("--analyze_test", action='store_true', default=False, dest="analyze_test")

parser.add_option("--print_log", action='store_true', default=False, dest="print_log")
parser.add_option("--pred_weight", type="float", default=0.3, dest="pred_weight")
parser.add_option("--no_road", action='store_false', default=True, dest="use_road_mask")

(options, args) = parser.parse_args()

pred_dir = options.pred_dir
pred_name = options.pred_name
save_path = options.save_path

path_raw = options.path_raw
path_train_set = options.path_train_set
path_cv_set = options.path_cv_set
path_test_set = options.path_test_set
path_road = options.path_road

analyze_all = options.analyze_all
analyze_train = options.analyze_train
analyze_CV = options.analyze_CV
analyze_test = options.analyze_test

pred_weight = options.pred_weight
print_log = options.print_log
use_road_mask = options.use_road_mask

pred_dir = pred_dir.strip('/') + '/'
save_path = save_path.strip('/') + '/'
assert not (save_path is None)
if not os.path.exists(save_path): os.makedirs(save_path)

save_name = pred_name.split('.')[0]

if analyze_all:

    h5f = h5py.File(path_raw, 'r')
    raw_image = np.array(h5f['scene'])
    h5f.close()
    
    if use_road_mask:
        road_mask = skimage.io.imread(path_road)
        name_postfix = ''
    else:
        road_mask = None
        name_postfix = '_noR'

    h5f = h5py.File(pred_dir + pred_name, 'r')
    pred_road = np.array(h5f['20170531'])
    h5f.close()

    if print_log:
        show_log_pred_with_raw(raw_image, pred_road, road_mask, pred_weight=pred_weight, figsize=(150,150), 
                               show_plot=False, save_path=save_path + save_name + '_all_' + str(pred_weight).replace('.', '_') + name_postfix + '_log.png')
    else:
        show_pred_prob_with_raw(raw_image, pred_road, road_mask, pred_weight=pred_weight, figsize=(150,150), 
                                show_plot=False, save_path=save_path + save_name + '_all_' + str(pred_weight).replace('.', '_') + name_postfix + '.png')    

if analyze_train:
    # Load training set
    train_set = h5py.File(path_train_set, 'r')
    train_raw_image = np.array(train_set['raw_image'])
    train_road_mask = np.array(train_set['road_mask'])
    train_set.close()

    h5f = h5py.File(pred_dir + pred_name, 'r')
    train_pred = np.array(h5f['train_pred'])
    h5f.close()

    gc.collect()
    
    if print_log:
        show_log_pred_with_raw(train_raw_image, train_pred, train_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                               show_plot=False, save_path=save_path + save_name + '_train_' + str(pred_weight).replace('.', '_') + '_log.png')
    else:
        show_pred_prob_with_raw(train_raw_image, train_pred, train_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                                show_plot=False, save_path=save_path + save_name + '_train_' + str(pred_weight).replace('.', '_') + '.png')
    plt.close()

gc.collect()

if analyze_CV:
    # Load cross-validation set
    CV_set = h5py.File(path_cv_set, 'r')
    CV_raw_image = np.array(CV_set['raw_image'])
    CV_road_mask = np.array(CV_set['road_mask'])
    CV_set.close()

    h5f = h5py.File(pred_dir + pred_name, 'r')
    CV_pred = np.array(h5f['CV_pred'])
    h5f.close()

    gc.collect()

    if print_log:
        show_log_pred_with_raw(CV_raw_image, CV_pred, CV_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                               show_plot=False, save_path=save_path + save_name + '_CV_' + str(pred_weight).replace('.', '_') + '_log.png')
    else:
        show_pred_prob_with_raw(CV_raw_image, CV_pred, CV_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                                show_plot=False, save_path=save_path + save_name + '_CV_' + str(pred_weight).replace('.', '_') + '.png')
    plt.close()

gc.collect()

if analyze_test:
    test_set = h5py.File(path_test_set, 'r')
    test_raw_image = np.array(test_set['raw_image'])
    test_road_mask = np.array(test_set['road_mask'])
    test_set.close()

    h5f = h5py.File(pred_dir + pred_name, 'r')
    test_pred = np.array(h5f['test_pred'])
    h5f.close()

    if print_log:
        show_log_pred_with_raw(test_raw_image, test_pred, test_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                               show_plot=False, save_path=save_path + save_name + '_CV_' + str(pred_weight).replace('.', '_') + '_log.png')
    else:
        show_pred_prob_with_raw(test_raw_image, test_pred, test_road_mask, pred_weight=pred_weight, figsize=(150,150), 
                                show_plot=False, save_path=save_path + save_name + '_CV_' + str(pred_weight).replace('.', '_') + '.png')
    plt.close()



