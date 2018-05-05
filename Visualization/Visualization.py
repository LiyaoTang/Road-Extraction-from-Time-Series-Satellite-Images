import numpy as np
import matplotlib.pyplot as plt

# pass in whole image (only raw) => display a subset of it
def show_raw_image(image, x = 0, y = 0, light=3.0, size=500, x_size=0, y_size=0, figsize=(20,20),
                   show_plot = True, save_path=None, close_plot=True):
    if x_size == 0 and y_size == 0:
        x_size = size
        y_size = size

    if size > 0:
        sub_image = image[[1,2,3],x:x+x_size,y:y+y_size]
    else:
        sub_image = image[[1,2,3]]

    sub_image = sub_image/10000*light  
    
    for img in sub_image:
        img[np.where(img<0)] = 0
        img[np.where(img>1)] = 1

    patch = np.array([sub_image[2].T, sub_image[1].T, sub_image[0].T]).T
    plt.figure(figsize=figsize)
    plt.imshow(patch)
    if not save_path is None:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    if close_plot:
        plt.close()

# pass in whole image (raw & road) => display a subset of it (size = -1 to show the whole)
def show_image_against_road(image, road, x = 0,y = 0, light=3.0, size=500, figsize=(20,20), BGR_axis=[1,2,3],
                            show_plot = True, show_raw=True, threshold = 0, save_path=None, close_plot=True):

    if size > 0:
        sub_road = road[x:x+size,y:y+size]
        sub_image = image[BGR_axis,x:x+size,y:y+size]
    else:
        sub_road = road
        sub_image = image[BGR_axis]

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
    if close_plot:
        plt.close()
    
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

    # change the pixel on pred_road to 'R'
    pred_road_index = np.where(sub_road > threshold)
    sub_image[2][pred_road_index] = sub_road[pred_road_index]
    sub_image[1][pred_road_index] = 0
    sub_image[0][pred_road_index] = 0
    
    # strenthen the 'B' channel of true roads
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
    plt.close()
    
    
def show_pred_prob_with_raw(image, road_prob, true_road=None, coord=(0,0), size=-1, light=3.0, pred_weight=0.3,
                            figsize=None, show_plot=True, save_path=None):

    sub_image = image[[1,2,3]] # BGR
    sub_image = sub_image/10000*light        
    
    for img in sub_image:
        img[np.where(img<0)] = 0
        img[np.where(img>1)] = 1
    
    # add into R channel
    sub_image[2] = sub_image[2] + road_prob*pred_weight
    sub_image[2][np.where(sub_image[2]>1)] = 1
    
    # strenthen the B channel of true roads
    if not (true_road is None):
        true_road_index = np.where(true_road == 1)
        sub_image[0][true_road_index] = 1

    if size > 0:
        sub_image = sub_image[:,coord[0]:coord[0]+size, coord[1]:coord[1]+size]

    patch = np.array([sub_image[2], sub_image[1], sub_image[0]]).transpose(1,2,0)

    if figsize:
        plt.figure(figsize=figsize)
    
    plt.imshow(patch)
    
    if not save_path is None:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
        plt.clf()
        plt.close()

def show_log_pred_with_raw(raw_imgae, pred, road_mask=None, light=3.0, pred_weight=0.3, figsize=None, show_plot=True, save_path=None):
    log_pred = -np.log(-pred + 1 + 1e-9)
    norm_log_pred = (log_pred - log_pred.min()) / (log_pred.max()-log_pred.min())
    show_pred_prob_with_raw(raw_imgae, norm_log_pred, road_mask, light=light, pred_weight=pred_weight, 
                            figsize=figsize, show_plot=show_plot, save_path=save_path)
