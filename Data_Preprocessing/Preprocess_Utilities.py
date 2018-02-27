import numpy as np
import matplotlib.pyplot as plt

# given raw_image & road_img (road mask), return: raw_img patches, road mask, road existence
def create_labelled_patches(raw_image, road_img, 
                            row_offset = 0, column_offset = 0, step = 28, minimum_road_mark = 5):
    image_patch = []
    road_patch = []
    road_existence = []

    i = row_offset
    while(i+step <= raw_image[0].shape[0]):
        j = column_offset
        while (j+step <= raw_image[0].shape[1]):
            cur_img_patch = raw_image[:,i:i+step, j:j+step]
            cur_road_patch = road_img[i:i+step, j:j+step]

            if (cur_img_patch != -9999).all() and (
                cur_road_patch.sum() >= minimum_road_mark or cur_road_patch.sum() == 0):

                image_patch.append(cur_img_patch)
                road_patch.append(cur_road_patch)
                road_existence.append(not cur_road_patch.sum() == 0)
            j += step
        i += step
    
    return image_patch, road_patch, road_existence