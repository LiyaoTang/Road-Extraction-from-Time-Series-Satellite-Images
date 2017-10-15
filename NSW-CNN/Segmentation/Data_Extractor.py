
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# top-left coordinate should be of shape (n, 2) => default to flatten out
def get_patches(raw_image, road_mask, topleft_coordinate, step, flatten = True):
    X = []
    Y = []
    
    if flatten:
        for coord in topleft_coordinate:
            X.append(raw_image[:, coord[0]:coord[0]+step, coord[1]:coord[1]+step].flatten())
            Y.append(road_mask[int(coord[0]+step/2), int(coord[1]+step/2)])
    else:
        for coord in topleft_coordinate:
            X.append(raw_image[:, coord[0]:coord[0]+step, coord[1]:coord[1]+step])
            Y.append(road_mask[int(coord[0]+step/2), int(coord[1]+step/2)])
        

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

