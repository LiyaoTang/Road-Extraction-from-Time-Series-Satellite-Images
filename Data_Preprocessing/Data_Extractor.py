
# coding: utf-8

# In[1]:

import numpy as np
np.random.seed(0)

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

    normalization = 'mean'
    
    def __init__(self, raw_image, road_mask, step, pos_topleft_coord, neg_topleft_coord, normalization='mean', encoding='binary'):

        # original image info
        self.raw_image = raw_image.copy()
        self.road_mask = road_mask.copy()
        
        self.band = raw_image.shape[0]
    
        # patch info        
        self.step = step

        # batch info
        self.encoding = encoding # ground truth encoding
        self.normalization = normalization # if need norm

        # pos-neg coordinate info
        self.pos_topleft_coord = np.array(pos_topleft_coord)
        np.random.shuffle (self.pos_topleft_coord)
            
        self.neg_topleft_coord = np.array(neg_topleft_coord)
        np.random.shuffle (self.neg_topleft_coord)
        
        self.pos_size = pos_topleft_coord.shape[0]
        self.neg_size = neg_topleft_coord.shape[0]

        # create topleft_coordinate using both pos & neg
        self.topleft_coordinate = []
        self.topleft_coordinate.extend (pos_topleft_coord)
        self.topleft_coordinate.extend (neg_topleft_coord)
        self.topleft_coordinate = np.array(self.topleft_coordinate)
        np.random.shuffle (self.topleft_coordinate)
        
        self.size = self.topleft_coordinate.shape[0]
        
        if normalization: self._cal_norm_param()

    def _cal_norm_param(self):
        if self.normalization == 'mean':
            mu = 0
            step = self.step
            # Careful! norm params not yet calculated
            for patch in self.iterate_raw_image_patches(norm = False):
                mu += patch
                assert (patch != -9999).all()
            mu = mu / self.size
            self.mu = mu.mean(axis=(1,2))

    def _get_raw_patch(self, coord, norm):
        patch  = self.raw_image[:, coord[0]:coord[0]+self.step, coord[1]:coord[1]+self.step]
        if norm:
            for channel_num in range(self.band):
                patch[channel_num] = patch[channel_num] - self.mu[channel_num]
        return patch

    def _get_patch_label(self, coord):
        label = self.road_mask[int(coord[0]+self.step/2), int(coord[1]+self.step/2)]
        if self.encoding == 'one-hot':
            one_hot_label = np.zeros(2)
            one_hot_label[label] = 1
            return one_hot_label
        return label

    def iterate_raw_image_patches (self, norm):
        for coord in self.topleft_coordinate:
            yield self._get_raw_patch(coord, norm)
                
    def iterate_raw_image_patches_with_coord (self, norm):
        for coord in self.topleft_coordinate:
            yield coord, self._get_raw_patch(coord, norm)

    def iterate_data (self, norm=True):
        for coord in self.topleft_coordinate:
            x = self._get_raw_patch(coord, norm)
            y = self._get_patch_label(coord)
            yield x, y

    def iterate_data_with_coord (self, norm=True):
        for coord in self.topleft_coordinate:
            x = self._get_raw_patch(coord, norm)
            y = self._get_patch_label(coord)
            yield coord, x, y


    def _get_patches_from_topleft_coord (self, coord_arr, num_of_patches, start_index, norm, wrap_around):
        X = []
        Y = []
        
        start = start_index
        end = start_index + num_of_patches
        compensation = 0
            
        # check overflow
        if end > coord_arr.shape[0]:
            compensation = end - coord_arr.shape[0]
            end = coord_arr.shape[0]
        
        # appending
        for idx in range(start, end):
            coord = coord_arr [idx]

            X.append(self._get_raw_patch(coord, norm))
            Y.append(self._get_patch_label(coord))

        start_index = end

        # wrap around
        if compensation > 0:
            np.random.shuffle (coord_arr)

            if wrap_around:
                end = compensation
                for idx in range(0, end):
                    coord = coord_arr [idx]
                    
                    X.append(self._get_raw_patch(coord, norm))
                    Y.append(self._get_patch_label(coord))

                start_index = end

            else:
                start_index = 0            

        return X, Y, start_index
    
    # top-left coordinate should be of shape (n, 2)
    def get_patches(self, batch_size, positive_num = 0, norm = True, wrap_around=True):
        X = []
        Y = []
        
        if positive_num > 0 and positive_num <= batch_size:
            # pos patches
            X_pos, Y_pos, self.pos_index = self._get_patches_from_topleft_coord (self.pos_topleft_coord,
                                                                     num_of_patches = positive_num,
                                                                     start_index = self.pos_index,
                                                                     norm = norm, 
                                                                     wrap_around=wrap_around)
            
            negative_num = batch_size - positive_num
            X_neg, Y_neg, self.neg_index = self._get_patches_from_topleft_coord (self.neg_topleft_coord,
                                                                     num_of_patches = negative_num,
                                                                     start_index = self.neg_index,
                                                                     norm = norm, 
                                                                     wrap_around=wrap_around)
            
            X.extend (X_pos)
            Y.extend (Y_pos)
            
            X.extend (X_neg)
            Y.extend (Y_neg)
        
        else: # sample batches randomly
            X, Y, self.index = self._get_patches_from_topleft_coord (self.topleft_coordinate,
                                                                      num_of_patches = batch_size,
                                                                      start_index = self.index,
                                                                      norm = norm,
                                                                      wrap_around=wrap_around)

        X = np.array(X)
        Y = np.array(Y)
        return X, Y


# class Cross_Val_Data(Data_Extractor):

#     def __init__(self, raw_image, road_mask, step, pos_topleft_coord, neg_topleft_coord, normalization='mean', encoding='binary'):

#         # not storing coordinates
#         super(Cross_Val_Data, self).__init__(raw_image, road_mask, step, None, None, normalization, encoding)

#         # iter through pos
#         for coord in pos_topleft_coord:

#     def _cal_norm_param(self):
#         if self.normalization == 'mean':
#             mu = 0
#             step = self.step
#             # Careful! norm params not yet calculated
#             for patch in self.iterate_raw_image_patches(norm = False):
#                 mu += patch
#                 assert (patch != -9999).all()
#             mu = mu / self.size
#             self.mu = mu.mean(axis=(1,2))

#     def get_cv_set(self, norm=True):


#     def iterate_raw_image_patches(self, norm=False):
#         