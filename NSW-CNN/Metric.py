
# coding: utf-8

# In[24]:


import numpy as np

class Metric:
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    size = 0
    
    def accumulate(self,pred, Y):
        self.true_neg  += np.logical_and(pred == Y, np.logical_not(Y)).sum()
        self.false_neg += np.logical_and(pred != Y, np.logical_not(Y)).sum()
        self.true_pos  += np.logical_and(pred == Y, Y).sum()
        self.false_pos += np.logical_and(pred != Y, Y).sum()
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


# In[ ]:




