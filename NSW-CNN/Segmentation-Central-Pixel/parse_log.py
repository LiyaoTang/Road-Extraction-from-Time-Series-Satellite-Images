# encoding: utf-8

import numpy as np
import sklearn as sk
import sklearn.linear_model as sklm
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
parser.add_option("--dir", dest="dir")
(options, args) = parser.parse_args()

record_list = []

log_dir = options.dir
for filename in os.listdir(log_dir):

    train_record = dict{'pos_r', 'neg_r'}
    test_record = dict{'pos_r', 'neg_r'}
    record = dict{'weight', 'pos', 'epoch', 'norm_T', 'reg_param', 'rand', 'avg_pre', 'bal_acc', 'AUC', 'train', 'test'}

    file = open(filename)
    log = file.read().split('\n')
    file.close()
    # m0_001_p8_e15_r1
    record['weight'] = (filename.find('weight') > 0)
    record['pos']    = int(filename.split('p')[-1].split('_')[0])
    record['epoch']  = int(filename.split('e')[-1].split('_')[0])
    record['rand']   = int(filename.split('r')[-1].split('_')[0])
    if filename.find('G') > 0:
        record['norm_T'] = 'std'

        l = filename.split('G')[-1].split('_')
        record['reg_param'] = float(l[0])
        if record['reg_param'] == 0:
            record['reg_param'] = float('.'.join([l[0], l[1]]))
    else:
        record['norm_T'] = 'mean-center'
        l = filename.split('m')[-1].split('_')
        record['reg_param'] = float(l[0])
        if record['reg_param'] == 0:
            record['reg_param'] = float('.'.join([l[0], l[1]]))

    train_cnt = 0
    for line in log:
        if line.startswith():
            train_cnt += 1
            if train_cnt == record['epoch']:
                record['avg_pre'] = 
                record['bal_acc'] = 
                record['AUC'] = 
        if 