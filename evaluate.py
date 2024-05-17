import math
import heapq
import numpy as np
import torch
import multiprocessing
from tqdm import tqdm

def evaluate_model(model, test_ratings, test_negatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation.
    Return: scores of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = test_ratings
    _testNegatives = test_negatives
    _K = K

    hits, ndcgs = [], []
    if num_thread > 1:  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
    else:  # Single thread
        for idx in range(len(_testRatings)):
            hr, ndcg = eval_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)
    return hits, ndcgs

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx].copy()
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    
    # Get prediction scores
    users = np.full(len(items), u, dtype='int32')
    users = torch.tensor(users).to(_model.device)
    items = torch.tensor(items).to(_model.device)

    _model.eval()
    with torch.no_grad():
        predictions = _model(users, items).cpu().numpy()
    
    map_item_score = {item: predictions[i] for i, item in enumerate(items)}
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = get_hit_ratio(ranklist, gtItem)
    ndcg = get_ndcg(ranklist, gtItem)
    return hr, ndcg

def get_hit_ratio(ranklist, gtItem):
    return 1 if gtItem in ranklist else 0

def get_ndcg(ranklist, gtItem):
    for i, item in enumerate(ranklist):
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0
