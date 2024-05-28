import numpy as np
import torch


def hit(ng_item, pred_items):
	if ng_item in pred_items:
		return 1
	return 0


def ndcg(ng_item, pred_items):
	if ng_item in pred_items:
		index = pred_items.index(ng_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k, device):
	HR, NDCG = [], []

	for user, item, label in test_loader:
		user = user.to(device)
		item = item.to(device)
  
		predictions = model(user, item)

		_, indices = torch.topk(predictions, top_k, dim=0)
		recommends = torch.take(
				item, indices).cpu().numpy().T.tolist()
  
  
		# adjust the shape of recommends
		# GMF: [[1,1,1,1]], MLP: [[1,1,1,1]]
		if isinstance(recommends[0], list):
			recommends = recommends[0]
   
   
		ng_item = item[0].item() # leave one-out evaluation has only one item per user
		hr_ = hit(ng_item, recommends)
		ndcg_ = ndcg(ng_item, recommends)
		HR.append(hr_)
		NDCG.append(ndcg_)
  
	return np.mean(HR), np.mean(NDCG)