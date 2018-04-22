
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
parser.add_option("--sliced_road_data", dest="sliced_road_data")
parser.add_option("--image_dir", dest="image_dir")
parser.add_option("--image_name", dest="image_name")
parser.add_option("--sliced_image_data", dest="sliced_image_data")
parser.add_option("--split_line_path", dest="split_line_path")
parser.add_option("--save_dir", dest="save_dir_path")

(options, args) = parser.parse_args()

size    = options.size

sliced_image_data = options.sliced_image_data
sliced_road_data  = options.sliced_road_data
split_line_path   = options.split_line_path

image_dir = options.image_dir
image_name = options.image_name
image_path = image_dir+image_name
rd_type = options.type

save_dir_path = options.save_dir_path.strip('/') + '/'

assert (sliced_image_data is None) or (sliced_road_data is None) or (sliced_image_data == sliced_road_data)


# read in splitted lines to split
h5f = h5py.File(split_line_path, 'r')
line_1 = list(h5f['line_cv_test'])
line_2 = list(h5f['line_train_cv'])
h5f.close()


def copy_road_into_slice(road_mask, height, width, line_1, line_2, base_func, cmp_func):

    road_mask_slice = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            img_x = base_func(i)

            if cmp_func(img_x, j):
                road_mask_slice[i,j] = road_mask[img_x,j]
                
    return road_mask_slice

def copy_image_into_slice(raw_image, height, width, line_1, line_2, base_func, cmp_func):

    raw_image_slice = np.zeros((7, height, width))-9999

    for i in range(height):
        for j in range(width):
            img_x = base_func(i)

            if cmp_func(img_x, j):
                raw_image_slice[:,i,j] = raw_image[:,img_x,j]
                
    return raw_image_slice

def copy_into_slice(raw_image, road_mask, height, width, line_1, line_2, base_func, cmp_func):

    road_mask_slice = np.zeros((height, width))
    raw_image_slice = np.zeros((7, height, width))-9999
    print(raw_image_slice.shape, road_mask_slice.shape)

    for i in range(height):
        for j in range(width):
            img_x = base_func(i)

            if cmp_func(img_x, j):
                road_mask_slice[i,j] = road_mask[img_x,j]
                raw_image_slice[:,i,j] = raw_image[:,img_x,j]
                
    return (raw_image_slice, road_mask_slice)

# load- / creat- ing the road mask & raw image

if not (rd_type is None): # not sliced

    assert sliced_road_data is None
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
    assert (save_dir_path == "../Data/090085/Road_Data/" + dir_name + '/')
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

else:
    assert not (sliced_road_data is None) # sliced

    h5f = h5py.File(sliced_road_data, 'r')
    road_mask_train = np.array(h5f['train']['road_mask'])
    road_mask_cv    = np.array(h5f['cv']['road_mask'])
    road_mask_test  = np.array(h5f['test']['road_mask'])
    h5f.close()

# read in scene
if not (image_path is None): # not sliced
    assert sliced_image_data is None 

    h5f = h5py.File(image_path)
    raw_image = np.array(h5f['scene'])
    h5f.close()

else:
    assert not (sliced_image_data is None) # sliced

    h5f = h5py.File(sliced_image_data, 'r')
    raw_image_train = np.array(h5f['train']['raw_image'])
    raw_image_cv    = np.array(h5f['cv']['raw_image'])
    raw_image_test  = np.array(h5f['test']['raw_image'])
    h5f.close()


if not (image_path is None) and not (rd_type is None): # slice the image & road
    print("slicing image and road")

    # train
    raw_image_train, road_mask_train = copy_into_slice(raw_image, combined_road_mask, 
                                                       height= int(combined_road_mask.shape[0]-min(line_2))+1,
                                                       width=combined_road_mask.shape[1], 
                                                       line_1 = line_1, line_2 = line_2,
                                                       base_func = lambda i: i+int(min(line_2)),
                                                       cmp_func = lambda img_x, j: img_x > line_2[j])
    # cv
    raw_image_cv, road_mask_cv = copy_into_slice(raw_image, combined_road_mask, 
                                                 height= int(combined_road_mask.shape[0]-min(line_2))+1,
                                                 width=combined_road_mask.shape[1], 
                                                 line_1 = line_1, line_2 = line_2,
                                                 base_func = lambda i: i+int(min(line_2)),
                                                 cmp_func = lambda x, j: line_1[j] < x and x < line_2[j])
    # test
    raw_image_test, road_mask_test = copy_into_slice(raw_image, combined_road_mask, 
                                                     height= int(max(line_1))+1,
                                                     width=combined_road_mask.shape[1],
                                                     line_1 = line_1, line_2 = line_2,
                                                     base_func = lambda i: i,
                                                     cmp_func = lambda img_x, j: img_x < line_1[j])


elif not (image_path is None): # slice the image
    print("slicing image")

    # train
    raw_image_train = copy_image_into_slice(raw_image, 
                                                       height= int(raw_image[0].shape[0]-min(line_2))+1,
                                                       width=raw_image[0].shape[1], 
                                                       line_1 = line_1, line_2 = line_2,
                                                       base_func = lambda i: i+int(min(line_2)),
                                                       cmp_func = lambda img_x, j: img_x > line_2[j])
    # cv
    raw_image_cv = copy_image_into_slice(raw_image, 
                                                 height= int(raw_image[0].shape[0]-min(line_2))+1,
                                                 width=raw_image[0].shape[1], 
                                                 line_1 = line_1, line_2 = line_2,
                                                 base_func = lambda i: i+int(min(line_2)),
                                                 cmp_func = lambda x, j: line_1[j] < x and x < line_2[j])
    # test
    raw_image_test = copy_image_into_slice(raw_image, 
                                                     height= int(max(line_1))+1,
                                                     width=raw_image[0].shape[1],
                                                     line_1 = line_1, line_2 = line_2,
                                                     base_func = lambda i: i,
                                                     cmp_func = lambda img_x, j: img_x < line_1[j])

elif not (rd_type is None): # slice the road
    print("slicing road")

    # train
    road_mask_train = copy_road_into_slice(combined_road_mask, 
                                           height= int(combined_road_mask.shape[0]-min(line_2))+1,
                                           width=combined_road_mask.shape[1], 
                                           line_1 = line_1, line_2 = line_2,
                                           base_func = lambda i: i+int(min(line_2)),
                                           cmp_func = lambda img_x, j: img_x > line_2[j])
    # cv
    road_mask_cv = copy_road_into_slice(combined_road_mask, 
                                         height= int(combined_road_mask.shape[0]-min(line_2))+1,
                                         width=combined_road_mask.shape[1], 
                                         line_1 = line_1, line_2 = line_2,
                                         base_func = lambda i: i+int(min(line_2)),
                                         cmp_func = lambda x, j: line_1[j] < x and x < line_2[j])
    # test
    road_mask_test = copy_road_into_slice(combined_road_mask, 
                                         height= int(max(line_1))+1,
                                         width=combined_road_mask.shape[1],
                                         line_1 = line_1, line_2 = line_2,
                                         base_func = lambda i: i,
                                         cmp_func = lambda img_x, j: img_x < line_1[j])


show_image_against_road(raw_image_train, road_mask_train, size=-1, figsize=(50,50),
                        show_plot=False, show_raw=False, close_plot=False, save_path=save_dir_path+image_name+'_train.png')

show_image_against_road(raw_image_cv, road_mask_cv, size=-1, figsize=(50,50),
                        show_plot=False, show_raw=False, close_plot=False, save_path=save_dir_path+image_name+'_cv.png')

show_image_against_road(raw_image_test, road_mask_test, size=-1, figsize=(50,50),
                        show_plot=False, show_raw=False, close_plot=False, save_path=save_dir_path+image_name+'_test.png')


divide = True
step = 16
threshold = 1

h5_path = save_dir_path+image_name
h5f = h5py.File(h5_path, 'w')

# Train
train_h5f = h5f['train']
create_segment_set_with_name(raw_image=raw_image_train, combined_road_mask=road_mask_train,
                             size=size, step=step, divide=divide,
                             is_pos_exmp=lambda rd_mask: (rd_mask==1).sum() > 64,
        is_valid_patch=lambda patch: ((patch==-9999).sum() / np.prod(np.array(patch.shape))) < threshold/100,
                             h5f=train_h5f)

# cv
cv_h5f = h5f['cv']
create_segment_set_with_name(raw_image=raw_image_cv, combined_road_mask=road_mask_cv,
                             size=size, step=step, divide=divide,
                             is_pos_exmp=lambda rd_mask: (rd_mask==1).sum() > 64,
        is_valid_patch=lambda patch: ((patch==-9999).sum() / np.prod(np.array(patch.shape))) < threshold/100,
                             h5f=train_h5f)

# Test
test_h5f = h5f['test']
create_segment_set_with_name(raw_image=raw_image_test, combined_road_mask=road_mask_test,
                             size=size, step=step, divide=divide,
                             is_pos_exmp=lambda rd_mask: (rd_mask==1).sum() > 64,
        is_valid_patch=lambda patch: ((patch==-9999).sum() / np.prod(np.array(patch.shape))) < threshold/100,
                             h5f=train_h5f)