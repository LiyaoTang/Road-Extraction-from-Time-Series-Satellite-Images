
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.io
import numpy as np
import h5py
import sys
import gc

from Data_Extractor import *


# In[2]:


path_train_set = "../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_8_train"
path_cv_set = "../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_topleft_coord_split_8_cv"

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

gc.collect()


# In[3]:


size = 8
norm = 'Gaussian'
Train_Data = Data_Extractor (train_raw_image, train_road_mask, size,
                             pos_topleft_coord = train_pos_topleft_coord,
                             neg_topleft_coord = train_neg_topleft_coord,
                             normalization = norm)

gc.collect()




def is_close_to_edge(coord, raw_img, size):
    dist = int(1.5*size)
    return (raw_img[:,coord[0]-dist:coord[0]+dist, coord[1]-dist:coord[1]+dist] == -9999).any()


#################################
# Training Set
#################################

edge_img = []
center_img = []
i = 0
for coord, patch, y in Train_Data.iterate_data_with_coord(norm=True):
    if is_close_to_edge(coord,  train_raw_image, size):
        edge_img.extend(patch)
    else:
        center_img.extend(patch)
    i += 1
    if i > 5000:
        gc.collect()
        i = 0

edge_img = np.array(edge_img)
gc.collect()
center_img = np.array(center_img)
gc.collect()


h5f = h5py.File('./Result/center_edge_img_train.h5', 'w')
h5f.create_dataset(name='edge', data=edge_img)
h5f.create_dataset(name='center', data=center_img)
h5f.close()


#################################
# Training Set
#################################

CV_Data = Data_Extractor (CV_raw_image, CV_road_mask, size,
                          pos_topleft_coord = CV_pos_topleft_coord,
                          neg_topleft_coord = CV_neg_topleft_coord,
                          normalization = norm)
edge_img = []
center_img = []
i = 0
for coord, patch, y in CV_Data.iterate_data_with_coord(norm=True):
    if is_close_to_edge(coord,  train_raw_image, size):
        edge_img.extend(patch)
    else:
        center_img.extend(patch)
    i += 1
    if i > 5000:
        gc.collect()
        i = 0

edge_img = np.array(edge_img)
gc.collect()
center_img = np.array(center_img)
gc.collect()

h5f = h5py.File('./Result/center_edge_img_cv.h5', 'w')
h5f.create_dataset(name='edge', data=edge_img)
h5f.create_dataset(name='center', data=center_img)
h5f.close()
