import math
import heapq
import multiprocessing
import numpy as np
from time import time
import torch

def evaluate_model(model, testRatings, testNegatives, K, num_thread, device):
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits, ndcgs = [],[]
    if num_thread > 1:  # 多线程
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.starmap(eval_one_rating, [(idx, device) for idx in range(len(_testRatings))])
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)

    # 单线程
    for idx in range(len(_testRatings)):
        hr, ndcg = eval_one_rating(idx, device)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)

def eval_one_rating(idx, device):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)

    # 获取预测分数
    map_item_score = {}
    users = torch.LongTensor([u] * len(items)).to(device)
    items_tensor = torch.LongTensor(items).to(device)

    _model.eval()
    with torch.no_grad():
        predictions = _model(users, items_tensor).squeeze().cpu().numpy()

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # 评估前 K 个物品
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    return int(gtItem in ranklist)

def getNDCG(ranklist, gtItem):
    if gtItem in ranklist:
        return math.log(2) / math.log(ranklist.index(gtItem) + 2)
    return 0
