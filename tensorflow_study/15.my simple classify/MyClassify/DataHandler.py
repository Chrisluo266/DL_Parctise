import numpy as np
import math

class DataHandler(object):
    def __init__(self,seed, total_size,class_num,batch_size):
        self.batch_size = batch_size
        self.total_size = total_size
        self.class_num = class_num
        rdm = np.random.RandomState(seed)
        self.datas = rdm.randn(total_size,2)
        self.labels = []
        for x,y in self.datas:
            distannce = math.floor(x*x+y*y)
            distannce = min(distannce,class_num - 1)
            self.labels.append(distannce)
        self.onehots = np.eye(class_num)[self.labels]

    def get_batch(self):
        epoch =  math.ceil(self.total_size / self.batch_size)
        current_batch_index = 0
        while(True):
            start_index = current_batch_index * self.batch_size
            end_index = min((current_batch_index + 1) * self.batch_size,self.total_size)
            yield self.datas[start_index:end_index] ,self.onehots[start_index:end_index]
            current_batch_index = (current_batch_index + 1) % epoch