
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import h5py
import sys
import scipy.io as sio
import skimage.io

sys.path.append("../Visualization/")

from optparse import OptionParser
from Preprocess_Utilities import *
from Visualization import *


parser = OptionParser()

parser.add_option("--size", type="int", default=128, dest="size")
parser.add_option("-t", "--type", dest="type")

(options, args) = parser.parse_args()

size = options.size
rd_type = options.type

rd_type = [int(i) for i in rd_type.split('-')]


# create road mask paths list
road_dir_path = '../Data/090085/Road_Data/'
road_type = np.array(["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "track", # 0-6
                      "residential", "service", "road", "living_street", # 7-10
                      "all_roads"]) # 11 
road_mask_path = np.char.add(road_dir_path, 
                             np.char.add(np.char.add(road_type, '/'), np.char.add(road_type, '.tif')))


# create path to save_dir
road_name = np.array(["motor", "trunk", "pri", "sec", "tert", "uncl", "track", # 0-6
                      "res", "serv", "road", "livi", # 7-10
                      ])
dir_name = '_'.join([road_name[i] for i in rd_type])
save_dir_path = "../Data/090085/Road_Data/" + dir_name + '/'
print(dir_name, save_dir_path)


# read in road mask
print('road mask loaded in ...')
road_img_list = []
for cur_path in road_mask_path:
    print(cur_path)
    road_img_list.append(skimage.io.imread(cur_path))

road_img_list = np.array(road_img_list)

# assert 0-1 coding
assert (np.logical_or(road_img_list == 1, road_img_list == 0)).all()

# combine the road mask
print("Used masks ... ")
combined_road_mask = 0
for i in rd_type:
    print(road_mask_path[i])
    combined_road_mask += road_img_list[i]
print(combined_road_mask.shape, (combined_road_mask > 1).any())

combined_road_mask[np.where(combined_road_mask > 1)] = 1
assert (np.logical_or(combined_road_mask == 1, combined_road_mask == 0)).all()
skimage.io.imsave(save_dir_path + 'road_mask.tif', combined_road_mask)


# read in splitted raw image & lines to split
data_dir = "../Data/090085/"

split_line_name = "img_split_lines.h5"
h5f = h5py.File(data_dir + split_line_name, 'r')
line_1 = list(h5f['line_cv_test'])
line_2 = list(h5f['line_train_cv'])
h5f.close()

img_data_name = "img_split.h5"
h5f = h5py.File(data_dir + img_data_name, 'r')
train_h5f = h5f['train']
raw_image_train = np.array(train_h5f['raw_image'])

cv_h5f = h5f['cv']
raw_image_cv = np.array(cv_h5f['raw_image'])

test_h5f = h5f['test']
raw_image_test = np.array(test_h5f['raw_image'])
h5f.close()



def copy_road_into_slice(road_mask, height, width, line_1, line_2, base_func, cmp_func):

    road_mask_slice = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            img_x = base_func(i)

            if cmp_func(img_x, j):
                road_mask_slice[i,j] = road_mask[img_x,j]
                
    return road_mask_slice


# In[ ]:


# train
road_mask_train = copy_road_into_slice(combined_road_mask, 
                                       height= int(combined_road_mask.shape[0]-min(line_2))+1,
                                       width=combined_road_mask.shape[1], 
                                       line_1 = line_1, line_2 = line_2,
                                       base_func = lambda i: i+int(min(line_2)),
                                       cmp_func = lambda img_x, j: img_x > line_2[j])

show_image_against_road(raw_image_train, road_mask_train, size=-1, figsize=(50,50),
                        show_plot=False, show_raw=False, close_plot=False, save_path=save_dir_path+'train.png')

# CV
road_mask_CV = copy_road_into_slice(combined_road_mask, 
                                     height= int(combined_road_mask.shape[0]-min(line_2))+1,
                                     width=combined_road_mask.shape[1], 
                                     line_1 = line_1, line_2 = line_2,
                                     base_func = lambda i: i+int(min(line_2)),
                                     cmp_func = lambda x, j: line_1[j] < x and x < line_2[j])


show_image_against_road(raw_image_cv, road_mask_cv, size=-1, figsize=(50,50),
                        show_plot=False, show_raw=False, close_plot=False, save_path=save_dir_path+'cv.png')


road_mask_test = copy_road_into_slice(combined_road_mask, 
                                     height= int(max(line_1))+1,
                                     width=combined_road_mask.shape[1],
                                     line_1 = line_1, line_2 = line_2,
                                     base_func = lambda i: i,
                                     cmp_func = lambda img_x, j: img_x < line_1[j])

show_image_against_road(raw_image_test, road_mask_test, size=-1, figsize=(50,50),
                        show_plot=False, show_raw=False, close_plot=False, save_path=save_dir_path+'test.png')



divide = True
step = 16
threshold = 1
name = "posneg_seg_coord_split_thr"+str(threshold)+"_"+str(size)+"_"+str(step) # posneg_topleft_coord_split_thr1_128_16
print(name)

# Train
create_segment_set_with_name(raw_image=raw_image_train, combined_road_mask=road_mask_train,
                             size=size, step=step, divide=divide, save_dir_path=save_dir_path,
                             is_pos_exmp=lambda rd_mask: (rd_mask==1).sum() > 64,
        is_valid_patch=lambda patch: ((patch==-9999).sum() / np.prod(np.array(patch.shape))) < threshold/100,
                             name = name+"_train")

# CV
create_segment_set_with_name(raw_image=raw_image_cv, combined_road_mask=road_mask_CV,
                             size=size, step=step, divide=divide, save_dir_path=save_dir_path,
                             is_pos_exmp=lambda rd_mask: (rd_mask==1).sum() > 64,
        is_valid_patch=lambda patch: ((patch==-9999).sum() / np.prod(np.array(patch.shape))) < threshold/100,
                             name = name+"_cv")

# Test
create_segment_set_with_name(raw_image=raw_image_test, combined_road_mask=road_mask_test,
                             size=size, step=step, divide=divide, save_dir_path=save_dir_path,
                             is_pos_exmp=lambda rd_mask: (rd_mask==1).sum() > 64,
        is_valid_patch=lambda patch: ((patch==-9999).sum() / np.prod(np.array(patch.shape))) < threshold/100,
                             name = name+"_test")