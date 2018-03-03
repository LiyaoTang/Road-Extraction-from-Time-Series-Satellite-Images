import numpy as np
import matplotlib.pyplot as plt
import h5py

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


def create_set_with_name(raw_image, combined_road_mask, name, step, divide, save_dir_path, save_img=True):
    if divide:
        # record the top-left coordinate of each possible patches & divide into pos/neg groups
        pos_topleft_coordinate = []
        neg_topleft_coordinate = []

        row_offset = 0
        while (row_offset + step <= raw_image.shape[1]):
            col_offset = 0
            while (col_offset + step <= raw_image.shape[2]):
                cur_img_patch = raw_image         [:,row_offset:row_offset+step, col_offset:col_offset+step]
                cur_road_mask = combined_road_mask[  row_offset:row_offset+step, col_offset:col_offset+step]

                if (cur_img_patch != -9999).all():
                    if cur_road_mask [int(step/2), int(step/2)] == 1: # positive example
                        pos_topleft_coordinate.append((row_offset, col_offset))
                    else: # negative example
                        neg_topleft_coordinate.append((row_offset, col_offset))
              
                col_offset += 1
            row_offset += 1

        pos_topleft_coordinate = np.array(pos_topleft_coordinate)
        neg_topleft_coordinate = np.array(neg_topleft_coordinate)
        print("pos coordinates' shape=", pos_topleft_coordinate.shape)
        print("neg coordinates' shape=", neg_topleft_coordinate.shape)

        # save set
        h5_path = save_dir_path + name
        h5f = h5py.File(h5_path, 'w')
        h5f.create_dataset(name='positive_example', data=pos_topleft_coordinate)
        h5f.create_dataset(name='negative_example', data=neg_topleft_coordinate)
        if save_img:
            h5f.create_dataset(name='raw_image', data=raw_image)
            h5f.create_dataset(name='road_mask', data=combined_road_mask)
        h5f.close()

    else:
        # record the top-left coordinate of each possible patches sequentially
        topleft_coordinate = []

        row_offset = 0
        while (row_offset + step <= raw_image.shape[1]):
            col_offset = 0
            while (col_offset + step <= raw_image.shape[2]):
                cur_img_patch = raw_image[:,row_offset:row_offset+step, col_offset:col_offset+step]

                if (cur_img_patch != -9999).all():
                    topleft_coordinate.append((row_offset, col_offset))

                col_offset += 1
            row_offset += 1

        topleft_coordinate = np.array(topleft_coordinate)
        print("coordinates' shape=", topleft_coordinate.shape)

        # save set
        h5_path = save_dir_path + name
        h5f = h5py.File(h5_path, 'w')
        h5f.create_dataset(name='topleft_coordinate', data=topleft_coordinate)
        if save_img:
            h5f.create_dataset(name='raw_image', data=raw_image)
            h5f.create_dataset(name='road_mask', data=combined_road_mask)
        h5f.close()
    print("saved into ", h5_path)