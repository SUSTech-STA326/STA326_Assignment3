import pandas as pd
import math

class Metrics:
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None
    
    @property
    def subjects(self):
        return self._subjects
    
    @subjects.setter
    def subjects(self, subjects):
        test_users, test_items, test_scores ,neg_users, neg_items, neg_scores = \
            subjects[0], subjects[1], subjects[2],subjects[3], subjects[4], subjects[5]

        # the golden set
        test = pd.DataFrame({
            'user': test_users,
            'test_item': test_items,
            'test_score': test_scores
        })

        full = pd.DataFrame({
            'user': neg_users + test_users,
            'item': neg_items + test_items,
            'score': neg_scores + test_scores
        })

        full = pd.merge(full, test, on=['user'], how='left')

        full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        full.sort_values(['user', 'rank'], inplace=True)
        self._subjects = full
        
        
    def cal_hit_ratio(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        return len(test_in_top_k) * 1.0 / full['user'].nunique()
    
    def cal_ndcg(self):
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank'] <= top_k]
        test_in_top_k = top_k[top_k['test_item'] == top_k['item']]
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: math.log(2) / math.log(1 + x))
        return test_in_top_k['ndcg'].sum() * 1.0 / full['user'].nunique()