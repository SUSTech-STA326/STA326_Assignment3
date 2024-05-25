import math
import heapq
import multiprocessing
import numpy as np
from time import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits, ndcgs = [], []
    if num_thread > 1:  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return hits, ndcgs
    # Single thread
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return hits, ndcgs

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    users = np.full(len(items), u, dtype='int32')
    item_indices = np.array(items)

    users_tensor = torch.tensor(users, dtype=torch.long, device=device)
    item_indices_tensor = torch.tensor(item_indices, dtype=torch.long, device=device)

    with torch.no_grad():
        predictions = _model(users_tensor, item_indices_tensor)

    map_item_score = {item: score.item() for item, score in zip(items, predictions)}

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return hr, ndcg

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i, item in enumerate(ranklist):
        if item == gtItem:
            return 1 / math.log2(i + 2)
    return 0