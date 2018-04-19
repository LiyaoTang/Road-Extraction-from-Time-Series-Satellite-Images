# coding: utf-8

import numpy as np
np.random.seed(0)

class Data_Extractor:
    
    def __init__(self, raw_image, road_mask, img_size, pos_topleft_coord, neg_topleft_coord, normalization='mean', encoding='binary'):

        # original image info (shallow assignment to reduce mem)
        self.raw_image = raw_image
        self.road_mask = road_mask
        
        self.band = raw_image.shape[0]
    
        # patch info        
        self.img_size = img_size

        # batch info
        self.encoding = encoding # ground truth encoding
        self.normalization = normalization # if need norm

        # pos-neg coordinate info (array shallow assignment to reduce mem)
        self.pos_topleft_coord = pos_topleft_coord
        np.random.shuffle (self.pos_topleft_coord)
            
        self.neg_topleft_coord = neg_topleft_coord
        np.random.shuffle (self.neg_topleft_coord)
        
        self.pos_size = pos_topleft_coord.shape[0]
        self.neg_size = neg_topleft_coord.shape[0]

        self.pos_index = 0
        self.neg_index = 0
        self.index = 0

        # create topleft_coordinate using both pos & neg
        self.topleft_coordinate = []
        self.topleft_coordinate.extend (pos_topleft_coord)
        self.topleft_coordinate.extend (neg_topleft_coord)
        self.topleft_coordinate = np.array(self.topleft_coordinate)
        np.random.shuffle (self.topleft_coordinate)
        
        self.size = self.topleft_coordinate.shape[0]
        
        if normalization: 
            self.normalization = normalization
            assert normalization in set(['mean', 'Gaussian'])
            self._cal_norm_param()

    def _cal_norm_param(self):
        mu = 0
        img_size = self.img_size
        # Careful! norm params not yet calculated
        for patch in self.iterate_raw_image_patches(norm = False):
            mu += patch[0]
            assert (patch != -9999).all()
        mu = (mu / self.size).mean(axis=(1,2))
        self.mu = mu
        print("mu = ", mu)

        if self.normalization == 'Gaussian':
            std = 0
            mu_ext = np.repeat(mu, [np.prod(patch[0][0].shape)]*patch[0].shape[0]).reshape(patch[0].shape)
            
            for patch in self.iterate_raw_image_patches(norm = False):
                std += ((patch[0]-mu_ext)**2).mean(axis=(-1,-2))
            std = np.sqrt(std / self.size)
            self.std = std
            print('std = ', std)



    def _get_raw_patch(self, coord, norm):
        patch  = self.raw_image[:, coord[0]:coord[0]+self.img_size, coord[1]:coord[1]+self.img_size].copy()
        if norm:
            if self.normalization == 'mean':
                for channel_num in range(self.band):
                    patch[channel_num] = patch[channel_num] - self.mu[channel_num]
            else:
                for channel_num in range(self.band):
                    patch[channel_num] = (patch[channel_num] - self.mu[channel_num]) / self.std[channel_num]
        return patch

    def _get_patch_label(self, coord):
        label = self.road_mask[int(coord[0]+self.img_size/2), int(coord[1]+self.img_size/2)].copy()
        if self.encoding == 'one-hot':
            one_hot_label = np.zeros(2)
            one_hot_label[label] = 1
            return one_hot_label
        return label

    def iterate_raw_image_patches (self, norm):
        for coord in self.topleft_coordinate:
            yield np.array([self._get_raw_patch(coord, norm)])
                
    def iterate_raw_image_patches_with_coord (self, norm):
        for coord in self.topleft_coordinate:
            yield coord, np.array([self._get_raw_patch(coord, norm)])

    def iterate_data (self, norm=True):
        for coord in self.topleft_coordinate:
            x = np.array([self._get_raw_patch(coord, norm)])
            y = np.array([self._get_patch_label(coord)])
            yield x, y

    def iterate_data_with_coord (self, norm=True):
        for coord in self.topleft_coordinate:
            x = np.array([self._get_raw_patch(coord, norm)])
            y = np.array([self._get_patch_label(coord)])
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


class FCN_Data_Extractor (Data_Extractor):

    def __init__(self, raw_image, road_mask, img_size, pos_topleft_coord, neg_topleft_coord, normalization='mean', encoding='one-hot'):

        super(FCN_Data_Extractor, self).__init__(raw_image, road_mask, img_size, pos_topleft_coord, neg_topleft_coord, normalization, encoding)

    def _cal_norm_param(self):
        img_size = self.img_size
        mu = np.zeros((self.band, img_size, img_size))
        pixel_cnt = np.zeros((self.band, img_size, img_size))

        pos_cnt = 0
        neg_cnt = 0

        # Careful! norm params & weight not yet calculated
        for patch, rd_map, _ in self.iterate_data(norm=False, weighted=False):
            patch = patch[0]
            valid_img_mask = np.where(patch != -9999)
            valid_rd_mask = np.where(patch[0] != -9999)

            pixel_cnt[valid_img_mask] = pixel_cnt[valid_img_mask] + 1
            mu[valid_img_mask] = mu[valid_img_mask] + patch[valid_img_mask]

            if self.encoding == 'one-hot':
                pos_cnt += rd_map[0,:,:,1][valid_rd_mask].sum()
                neg_cnt += rd_map[0,:,:,0][valid_rd_mask].sum()
            else:
                pos_cnt += rd_map[valid_rd_mask].sum()
                neg_cnt += -(rd_map[valid_rd_mask]-1).sum()
    
        mu = mu.sum(axis=(1,2)) / pixel_cnt.sum(axis=(1,2))
        self.mu = mu
        self.neg_weight = pos_cnt / (pos_cnt+neg_cnt)
        self.pos_weight = neg_cnt / (pos_cnt+neg_cnt)

        assert (pos_cnt+neg_cnt) == pixel_cnt[0].sum()

        print("mu = ", mu)
        print("class weight [neg= %f, pos= %f]"%(self.neg_weight, self.pos_weight))

        if self.normalization == 'Gaussian':
            std = np.zeros((self.band, img_size, img_size))
            mu_ext = np.repeat(mu, [np.prod(patch[0].shape)]*patch.shape[0]).reshape(patch.shape)

            for patch in self.iterate_raw_image_patches(norm = False):
                patch = patch[0]
                valid_mask = np.where(patch != -9999)
                
                std[valid_mask] = std[valid_mask] + ((patch-mu_ext)**2)[valid_mask]
            
            std = np.sqrt(std.sum(axis=(1,2)) / pixel_cnt.sum(axis=(1,2)))
            self.std = std
            print('std = ', std)


    def _get_raw_patch(self, coord, norm):
        patch  = self.raw_image[:, coord[0]:coord[0]+self.img_size, coord[1]:coord[1]+self.img_size].copy()
        invalid_mask = np.where(patch == -9999)

        if norm:
            if self.normalization == 'mean':
                for channel_num in range(self.band):
                    patch[channel_num] = patch[channel_num] - self.mu[channel_num]
            elif self.normalization == 'Gaussian':
                for channel_num in range(self.band):
                    patch[channel_num] = (patch[channel_num] - self.mu[channel_num]) / self.std[channel_num]
        patch[invalid_mask] = 0
        return patch


    def _get_patch_label(self, coord):
        label = self.road_mask[coord[0]:coord[0]+self.img_size, coord[1]:coord[1]+self.img_size].copy()
        if self.encoding == 'one-hot':
            one_hot_label = np.zeros((self.img_size, self.img_size, 2))
            one_hot_label[:, :, 0][np.where(label == 0)] = 1
            one_hot_label[:, :, 1][np.where(label == 1)] = 1
            return one_hot_label
        return label

    # def _get_weight(self, X, Y):
    #     W = Y.copy()
    #     for idx in range(len(W)):
            


    def get_patches(self, batch_size, positive_num = 0, norm = True, wrap_around=True, weighted=True):
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
        W = Y.copy()
    
        if weighted:
            W[:,:,:,0] *= self.neg_weight
            W[:,:,:,1] *= self.pos_weight
            
        return X, Y, W.sum(axis=-1)


    def iterate_data (self, norm=True, weighted=True):
        for coord in self.topleft_coordinate:
            x = np.array([self._get_raw_patch(coord, norm)])
            y = np.array([self._get_patch_label(coord)])
            weight = y.copy()

            if weighted:
                weight[:,:,:,0] *= self.neg_weight
                weight[:,:,:,1] *= self.pos_weight

            yield x, y, weight.sum(axis=-1)


    def iterate_data_with_coord (self, norm=True, weighted=True):
        for coord in self.topleft_coordinate:
            x = np.array([self._get_raw_patch(coord, norm)])
            y = np.array([self._get_patch_label(coord)])
            weight = y.copy()

            if weighted and self.encoding == 'one-hot':
                weight[:,:,:,0] *= self.neg_weight
                weight[:,:,:,1] *= self.pos_weight
            
            yield coord, x, y, weight.sum(axis=-1)