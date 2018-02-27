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


# pass in whole image (raw & road) => display a subset of it
def show_image_against_road(image, road, x = 0,y = 0, light=3.0, size=500, figsize=(20,20),
                            show_plot = True, show_raw=True, threshold = 0, save_path=None):

    if size > 0:
        sub_road = road[x:x+size,y:y+size]
        sub_image = image[[1,2,3],x:x+size,y:y+size]
    else:
        sub_road = road
        sub_image = image[[1,2,3]]

    sub_road[np.where(sub_road == 255)] = 1        
    sub_image = sub_image/10000*light        
    
    for img in sub_image:
        img[np.where(img<0)] = 0
        img[np.where(img>1)] = 1
    
    if show_plot and show_raw:
        patch = np.array([sub_image[2].T, sub_image[1].T, sub_image[0].T]).T
        plt.figure(figsize=figsize)
        plt.imshow(patch)
        plt.show()
        plt.clf()

    sub_image[2][np.where(sub_road > threshold)] = 1
    sub_image[1][np.where(sub_road > threshold)] = 0
    sub_image[0][np.where(sub_road > threshold)] = 0

    patch = np.array([sub_image[2].T, sub_image[1].T, sub_image[0].T]).T
    plt.figure(figsize=figsize)
    plt.imshow(patch)
    if not save_path is None:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.clf()
    
# pass in whole image (raw & road) => display a subset of it
def show_pred_road_against_raw(image, pred_road, true_road=None, light=3.0,
                               figsize=(20,20), show_plot=True, show_raw=False, threshold = 0, save_path=None):

    sub_road = pred_road
    sub_image = image[[1,2,3]]

    sub_road[np.where(sub_road == 255)] = 1        
    sub_image = sub_image/10000*light        
    
    for img in sub_image:
        img[np.where(img<0)] = 0
        img[np.where(img>1)] = 1
    
    if show_plot and show_raw:
        patch = np.array([sub_image[2].T, sub_image[1].T, sub_image[0].T]).T
        plt.figure(figsize=figsize)
        plt.imshow(patch)
        plt.show()
        plt.clf()

    pred_road_index = np.where(sub_road > threshold)
    sub_image[2][pred_road_index] = sub_road[pred_road_index]
    sub_image[1][pred_road_index] = 0
    sub_image[0][pred_road_index] = 0
        
    if not (true_road is None):
        true_road_index = np.where(true_road == 1)
        sub_image[0][true_road_index] = 1

    patch = np.array([sub_image[2].T, sub_image[1].T, sub_image[0].T]).T
    plt.figure(figsize=figsize)
    plt.imshow(patch)
    if not save_path is None:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.clf()
