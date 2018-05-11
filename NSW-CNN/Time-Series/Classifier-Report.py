
# coding: utf-8

# In[1]:
# "Incep_1-32|1-64|1-128_G_weight_bn_p0_e20_r0"
# "Incep_3-32|3-64|3-128_G_weight_p0_e20_r0"
# "Incep_3-32;1-32|3-64;1-64_m_weight_bn_p0_e20_r0"
# "Incep_1-32;3-32|1-64;3-64|1-128;3-128_G_weight_p0_e20_r0"

# sk-SGD_weight_m5_p1_e15_r1
# sk-SGD_weight_p0_e15_r0

# sk-SGD_weight_G0.0001_p0_e15_rNone 
# sk-SGD_G0_0001_p32_e15_r0 
# sk-SGD_G0_001_p16_e15_r1 
# sk-SGD_G0_001_p8_e15_r0 
# sk-SGD_G0_001_p0_e15_r0   

import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--model_dir", dest="path_model_dir")
parser.add_option("--model_name", dest="model_name")
parser.add_option("--gpu", default="", dest="gpu")
(options, args) = parser.parse_args()

path_model_dir = options.path_model_dir
model_name     = options.model_name
gpu            = options.gpu

if not path_model_dir.endswith('/'): path_model_dir = path_model_dir + '/'
model_name = model_name.replace('.', '_')

# restrict to single gpu
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu


import numpy as np
import tensorflow as tf
import sklearn as sk
import sklearn.linear_model as sklm
import sklearn.metrics as skmt
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.io
import h5py
import sys
import gc

sys.path.append('../Metric/')
sys.path.append('../../Visualization/')
sys.path.append('../../Data_Preprocessing//')
from Metric import *
from Visualization import *
from Data_Extractor import *
from Restored_Classifier import *
from sklearn.externals import joblib



if 'sk-SGD' in model_name:
    classifier_type = 'LR'

    path_train_set = "../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_8_train"
    path_cv_set    = "../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_8_cv"
    path_test_set  = "../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_8_test"

    size = 8

else:
    assert 'Incep' in model_name or 'FCN' in model_name
    classifier_type = 'FCN'

    path_train_set = "../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_seg_coord_split_thr1_128_16_train"
    path_cv_set    = "../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_seg_coord_split_thr1_128_16_cv"
    path_test_set  = "../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_seg_coord_split_thr1_128_16_test"

    size = 128

use_norm = True
if model_name.find("_G") > 0:
    norm = "Gaussian"
elif model_name.find("_m") > 0:
    norm = "mean"
else:
    norm = None
    use_norm = False

print(norm, size, classifier_type, 'from', model_name)

# re-load classifier
classifier = Classifier(path_model_dir, model_name, classifier_type)


def evaluate_on_set(classifier, data_extractor, use_norm):
    if classifier.classifier_type == 'LR':

        metric = Metric_Record()
        for x, y in data_extractor.iterate_data(norm=use_norm):

            pred_prob = classifier.predict(x)
            pred = int(pred_prob > 0.5)
            metric.accumulate(Y=y, pred=pred, pred_prob=pred_prob)

        # calculate value
        metric.print_info()
        balanced_acc = metric.get_balanced_acc()
        AUC_score = skmt.roc_auc_score(metric.y_true, metric.pred_prob)
        avg_precision_score = skmt.average_precision_score(metric.y_true, metric.pred_prob)

        print("balanced_acc = ", balanced_acc, "AUC = ", AUC_score, "avg_precision = ", avg_precision_score)
        sys.stdout.flush()

    else:
        assert classifier.classifier_type == 'FCN'
        class_output = 2

        xen_list = []
        metric = Metric_Record()
        for batch_x, batch_y, batch_w in data_extractor.iterate_data(norm=use_norm):

            pred_prob = classifier.predict(batch_x)
            xen_list.append(classifier.get_mean_cross_xen(batch_x, batch_y, batch_w))

            metric.accumulate(Y         = np.array(batch_y.reshape(-1,class_output)[:,1]>0.5, dtype=int), 
                              pred      = np.array(pred_prob.reshape(-1,class_output)[:,1]>0.5, dtype=int), 
                              pred_prob = pred_prob.reshape(-1,class_output)[:,1])
        # print(np.array(metric.y_true).flatten().shape, np.array(metric.pred_prob).flatten().shape)

        # calculate value
        metric.print_info()
        balanced_acc = metric.get_balanced_acc()
        AUC_score = skmt.roc_auc_score(np.array(metric.y_true).flatten(), np.array(metric.pred_prob).flatten())
        avg_precision_score = skmt.average_precision_score(np.array(metric.y_true).flatten(), np.array(metric.pred_prob).flatten())

        print("balanced_acc = ", balanced_acc, "AUC = ", AUC_score, "avg_precision = ", avg_precision_score)
        print("xen = ", sum(xen_list)/len(xen_list))
        sys.stdout.flush()

    metric = 0
    xen_list = []
    gc.collect()

for path_data_set in [path_train_set, path_cv_set, path_test_set]:

    print("On", path_data_set.split('_')[-1], 'set')

    data_set = h5py.File(path_data_set, 'r')
    pos_topleft_coord = np.array(data_set['positive_example'])
    neg_topleft_coord = np.array(data_set['negative_example'])
    raw_image = np.array(data_set['raw_image'])
    road_mask = np.array(data_set['road_mask'])
    data_set.close()

    if classifier.classifier_type == 'LR':

        data_extractor = Data_Extractor (raw_image, road_mask, size,
                                         pos_topleft_coord = pos_topleft_coord,
                                         neg_topleft_coord = neg_topleft_coord,
                                         normalization = norm)

    else:
        assert classifier.classifier_type == 'FCN'

        data_extractor = FCN_Data_Extractor (raw_image, road_mask, size,
                                             pos_topleft_coord = pos_topleft_coord,
                                             neg_topleft_coord = neg_topleft_coord,
                                             normalization = norm)
    gc.collect()

    evaluate_on_set(classifier=classifier, data_extractor=data_extractor, use_norm=use_norm)


    # free
    pos_topleft_coord = 0
    neg_topleft_coord = 0
    data_extractor = 0
    raw_image = 0
    road_mask = 0
    gc.collect()
