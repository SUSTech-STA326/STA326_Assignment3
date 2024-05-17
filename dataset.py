from torch.utils.data import Dataset
import scipy.sparse as sp
import random
import pickle
import numpy as np

class RatingDataset(Dataset):
    def __init__(self, rating_mat, negative_list, user_num, item_num, negative_num=4):
        self.user_num = user_num
        self.item_num = item_num

        # for positive samples
        row_idx, col_idx = rating_mat.nonzero()
        self.user_ids = row_idx.tolist()
        self.item_ids = col_idx.tolist()
        # self.ratings = rating_mat[row_idx, col_idx].toarray().astype(np.float32)
        self.labels = np.ones(len(row_idx)).tolist()

        # print(self.user_num,len(negative_list))
        # extend list for negative samples
        for i in range(self.user_num):
            negatives_items = random.sample(negative_list[i],negative_num)
            self.user_ids.extend([i,i,i,i])
            self.item_ids.extend(negatives_items)
            self.labels.extend([0,0,0,0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # get user id and item id
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        labels = self.labels[idx]

        # create one hot vector (may not necessary, here I comment it because the embedding only need indices)
        # user_vec = torch.zeros(self.user_num,dtype = torch.float32)
        # item_vec = torch.zeros(self.item_num,dtype = torch.float32)
        # user_vec[user_id] = 1
        # item_vec[item_id] = 1
        # return user_vec, item_vec, labels
        return user_id, item_id, labels

def load_rating_file_as_sparse(filename):
    '''
    for train_rating file
    '''
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        for line in f:
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            num_users = max(num_users, user)
            num_items = max(num_items, item)

    num_users += 1
    num_items += 1
    mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    with open(filename, "r") as f:
        for line in f:
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if rating > 0:
                mat[user, item] = rating
    # print(num_users,num_items)
    return mat.tocsr(), num_users, num_items

def load_negative_file(filename):
    '''
    for test_negative file
    '''
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1: ]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList

def load_rating_file_as_list(filename):
    '''
    for test_rating file
    '''
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList

def save_preprocessed_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_preprocessed_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
