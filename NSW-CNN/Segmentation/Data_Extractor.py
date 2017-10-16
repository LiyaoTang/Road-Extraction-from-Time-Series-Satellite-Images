
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


# top-left coordinate should be of shape (n, 2)
def get_patches (raw_image=None, road_mask=None, topleft_coordinate=None, batch_size=0, positive_num=0, step=None):
    X = []
    Y = []
    
    if batch_size == 0: # sample patches according to the given topleft_coordinate
        for coord in topleft_coordinate:
            X.append(raw_image[:, coord[0]:coord[0]+step, coord[1]:coord[1]+step].flatten())
            Y.append(road_mask[int(coord[0]+step/2), int(coord[1]+step/2)])

    elif positive_num > 0: # random sample batches containing at least specified num of positive example
        rand_coord_idx = np.random.randint(topleft_coordinate.shape[0], size=batch_size)
        for coord in topleft_coordinate[rand_coord_idx]:
            Y.append(road_mask[int(coord[0]+step/2), int(coord[1]+step/2)])
        while sum(Y) < positive_num:
            Y = []
            rand_coord_idx = np.random.randint(topleft_coordinate.shape[0], size=batch_size)
            for coord in topleft_coordinate[rand_coord_idx]:
                Y.append(road_mask[int(coord[0]+step/2), int(coord[1]+step/2)])
        
        for coord in topleft_coordinate[rand_coord_idx]:
            X.append(raw_image[:, coord[0]:coord[0]+step, coord[1]:coord[1]+step].flatten())

    else: # random sample batches
        rand_coord_idx = np.random.randint(topleft_coordinate.shape[0], size=batch_size)
        for coord in topleft_coordinate[rand_coord_idx]:
            X.append(raw_image[:, coord[0]:coord[0]+step, coord[1]:coord[1]+step].flatten())
            Y.append(road_mask[int(coord[0]+step/2), int(coord[1]+step/2)])

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# In[ ]:




