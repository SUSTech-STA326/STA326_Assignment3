from typing import List
import scipy.sparse as sp
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
class Data:
    def __init__(self, path: str) -> None:
        self.train_data, self.train_mat = self.load_train_data(path + '.train.rating')
        self.num_user, self.num_item = self.train_mat.shape
        self.test_data = self.load_test_data(path+'.test.negative')  
    
    def load_train_data(self, filename: str):
        train_data = pd.read_csv(filename, sep='\t', header=None, names=['user', 'item'], usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
        num_user, num_item = train_data['user'].max() + 1, train_data['item'].max() + 1
        train_data = train_data.values.tolist()
        
        train_mat = sp.dok_matrix((num_user,num_item), dtype=np.float32)
        for i in train_data:
            train_mat[i[0],i[1]] = 1.0
        return train_data, train_mat
    
    def load_test_data(self, filename: str):
        test_data = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line is not None and line != '':
                a = line.split('\t')
                u = eval(a[0])[0]
                test_data.append([u, eval(a[0])[1]])
                for i in a[1:]:
                    test_data.append([u, int(i)])
                line = f.readline()
        return test_data

class Data2dataset(Dataset):
    def __init__(self, data: Data, num_ng: int, istrain: bool ) -> None:
        super(Data2dataset, self).__init__()
        self.train_f = data.train_data
        self.train_mat = data.train_mat
        self.test_f = data.test_data
        self.test_l = [0 for _ in range(len(data.test_data))] 
        self.istrain = istrain
        self.num_ng = num_ng
        self.num_item = data.num_item
        
    def __len__(self) -> int:
        return (self.num_ng + 1) * len(self.train_f) if self.istrain else len(self.test_f)
        
    def __getitem__(self, index):
        features = self.train_f_full if self.istrain else self.test_f
        labels = self.train_l if self.istrain else self.test_l
        
        user = features[index][0]
        item = features[index][1]
        label = labels[index]
        return user, item ,label
    
    def ng_sample(self):
        assert self.istrain, 'no need to sampling when testing'
        
        self.train_f_ng = []
        for i in self.train_f:
            u = i[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u,j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.train_f_ng.append([u,j])
        
        labels_p = [1 for _ in range(len(self.train_f))]
        labels_n = [0 for _ in range(len(self.train_f_ng))]
        
        self.train_f_full = self.train_f + self.train_f_ng
        self.train_l = labels_p + labels_n    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    # def load_rating_file_as_list(self, filename: str) -> List[List[int]]:
    #     rating_list = []
    #     with open(filename,'r') as f:
    #         line = f.readline()
    #         while line is not None and line != '':
    #               a = line.split('\t')
    #               user, item = int(a[0]), int(a[1])
    #               rating_list.append([user,item])
    #               line = f.readline()    
    #     return rating_list
    
    # def load_negative_file(self, filename: str) -> List[List[int]]:
    #     negative_list = []
    #     with open(filename, 'r') as f:
    #         line = f.readline()
    #         while line is not None and line != '':
    #             a = line.split('\t')
    #             negative_item = []
    #             for i in a[1:]:
    #                 negative_item.append(int(i))
    #             negative_list.append(negative_item)
    #             line = f.readline()
    #     return negative_list
       
    
    # def load_rating_file_as_matrix(self, filename: str):
    #     num_user, num_item = 0,0
    #     with open(filename,'r') as f:
    #         line = f.readline()
    #         while line is not None and line != '':
    #             a = line.split('\t')
    #             u, i = int(a[0]), int(a[1])
    #             num_user = max(num_user, u)
    #             num_item = max(num_item, i)
    #             line = f.readline()
    #     matrix = sp.dok_matrix(num_user+1,num_item+1)
    #     with open(filename,'r') as f:
    #         line = f.readline()
    #         while line is not None and line != '':
    #             a = line.split('\t')
    #             u, i, rating = int(a[0]), int(a[1]), int(a[2])
    #             matrix[u,i] = 1 if rating > 0 else 0 
    #             line = f.readline()  
    #     return matrix