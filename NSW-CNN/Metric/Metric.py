
# coding: utf-8

# In[1]:


import numpy as np

class Metric:
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    size = 0
    
    __index = {'TP':[], 'FP':[], 'TN':[], 'FN':[]}
    __record_index = False

    def __init__(self, record_index = False):
        self.__index = {'TP':[], 'FP':[], 'TN':[], 'FN':[]}
        self.__record_index = record_index
        
    def accumulate(self, pred, Y):        
        TP_arr = np.logical_and(pred, Y)
        FP_arr = np.logical_and(pred, np.logical_not(Y))
        TN_arr = np.logical_and(np.logical_not(pred), np.logical_not(Y))
        FN_arr = np.logical_and(np.logical_not(pred), Y)
        
        if self.__record_index:            
            self.__index['TP'].extend(self.size + np.where(TP_arr)[0])
            self.__index['FP'].extend(self.size + np.where(FP_arr)[0])
            self.__index['TN'].extend(self.size + np.where(TN_arr)[0])
            self.__index['FN'].extend(self.size + np.where(FN_arr)[0])
        
        self.true_pos  += TP_arr.sum()
        self.false_pos += FP_arr.sum()
        self.true_neg  += TN_arr.sum()
        self.false_neg += FN_arr.sum()
        self.size += Y.size
            
    def cal_metric(self):
        true_pos  = self.true_pos
        false_pos = self.false_pos
        true_neg  = self.true_neg
        false_neg = self.false_neg
        
        result = {}
        
        try: # metric for positive
            pos_recall = true_pos / (true_pos + false_neg)
            result['pos_recall'] = pos_recall

            pos_precision = true_pos / (true_pos + false_pos)
            result['pos_precision'] = pos_precision
    
            pos_F1 = 2*(pos_recall*pos_precision) / (pos_recall+pos_precision)
            result['pos_F1'] = pos_F1
        except:
            print("Error in F1 score for Positive")            
        
        try: # metric for negative
            neg_precision = true_neg / (true_neg + false_neg)
            result['neg_precision'] = neg_precision
            
            neg_recall = true_neg / (true_neg + false_pos)
            result['neg_recall'] = neg_recall

            neg_F1 = 2*(neg_recall*neg_precision) / (neg_recall+neg_precision)
            result['neg_F1'] = neg_F1
        except:
            print("Error in F1 score for Negative") 
            
        try: # accuracy
            accuracy = (true_pos + true_neg) / self.size
            result['accuracy'] = accuracy
        except:
            print("Error in accuracy")
            
        return result
    
    def print_info(self):
        print("%-9s = %d\n%-9s = %d\n%-9s = %d\n%-9s = %d\nsize = %d"
                         % ('true_pos', self.true_pos, 'false_pos', self.false_pos,
                            'true_neg', self.true_neg, 'false_neg', self.false_neg, self.size))
        
        result = self.cal_metric()
        for key in result.keys():
            print("%-13s = %s" % (key, result[key]))
            
    def get_index(self, key):
        if self.__record_index:
            return np.array(self.__index[key])
        else:
            print("Index for TP/FP/TN/FN not recorded")


# In[ ]:




