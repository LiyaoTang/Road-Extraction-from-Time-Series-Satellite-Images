
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class Data_Extractor:
    raw_image = None
    road_mask = None
    
    topleft_coordinate = []
    pos_topleft_coord = []
    neg_topleft_coord = []
 
    index = 0
    pos_index = 0
    neg_index = 0
    
    step = 0
    
    pos_size = 0
    neg_size = 0
    size = 0
    
    def __init__(self, raw_image, road_mask, step, pos_topleft_coord, neg_topleft_coord):
        self.raw_image = raw_image
        self.road_mask = road_mask
        
        self.pos_topleft_coord = np.array(pos_topleft_coord)
        np.random.shuffle (self.pos_topleft_coord)
            
        self.neg_topleft_coord = np.array(neg_topleft_coord)
        np.random.shuffle (self.neg_topleft_coord)
            
        self.topleft_coordinate.extend (pos_topleft_coord)
        self.topleft_coordinate.extend (neg_topleft_coord)
        self.topleft_coordinate = np.array(self.topleft_coordinate)
        np.random.shuffle (self.topleft_coordinate)
        
        self.step = step
        
        self.pos_size = pos_topleft_coord.shape[0]
        self.neg_size = neg_topleft_coord.shape[0]
        self.size = self.topleft_coordinate.shape[0]
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.topleft_coordinate.shape[0]:
            self.index = 0
            raise StopIteration
        
        coord = self.topleft_coordinate [self.index]        
        self.index = self.index + 1
        
        return self.raw_image[:, coord[0]:coord[0]+self.step, coord[1]:coord[1]+self.step].flatten()
    
    def __get_patches_from_topleft_coord (self, corrd_arr, num_of_patches, start_index):
        X = []
        Y = []
        
        raw_image = self.raw_image
        road_mask = self.road_mask
        
        step = self.step
        
        start = start_index
        end = start_index + num_of_patches
        compensation = 0
            
        if end > corrd_arr.shape[0]:
            compensation = end - corrd_arr.shape[0]
            end = corrd_arr.shape[0]
                
        for idx in range(start, end):
            coord = corrd_arr [idx]
            X.append(raw_image[:, coord[0]:coord[0]+step, coord[1]:coord[1]+step].flatten())
            Y.append(road_mask[int(coord[0]+step/2), int(coord[1]+step/2)])

        if compensation > 0:
            np.random.shuffle (corrd_arr)

            end = compensation
            for idx in range(0, end):
                coord = corrd_arr [idx]
                X.append(raw_image[:, coord[0]:coord[0]+step, coord[1]:coord[1]+step].flatten())
                Y.append(road_mask[int(coord[0]+step/2), int(coord[1]+step/2)])
        start_index = end
            
        return X, Y, start_index
    
    # top-left coordinate should be of shape (n, 2)
    def get_patches(self, batch_size, positive_num = 0):
        X = []
        Y = []
        
        if positive_num > 0 and positive_num <= batch_size:
            # pos patches
            X_pos, Y_pos, self.pos_index = self.__get_patches_from_topleft_coord (self.pos_topleft_coord,
                                                                     num_of_patches = positive_num,
                                                                     start_index = self.pos_index)
            
            negative_num = batch_size - positive_num
            X_neg, Y_neg, self.neg_index = self.__get_patches_from_topleft_coord (self.neg_topleft_coord,
                                                                     num_of_patches = negative_num,
                                                                     start_index = self.neg_index)
            
            X.extend (X_pos)
            Y.extend (Y_pos)
            
            X.extend (X_neg)
            Y.extend (Y_neg)
        
        else: # sample batches randomly
            X, Y, self.index = self.__get_patches_from_topleft_coord (self.topleft_coordinate,
                                                                      num_of_patches = batch_size,
                                                                      start_index = self.index)

        X = np.array(X)
        Y = np.array(Y)
        return X, Y


# In[ ]:




