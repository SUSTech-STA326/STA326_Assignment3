import numpy as np
import torch


def hit(gt_item, pred_items):
	if len(gt_item) == 0:
		return 0
	# 判断小张量中的值是否在大张量中
	isin = torch.isin(gt_item, pred_items)
	# 统计存在的值的数量
	count = isin.sum().item()
	return count/len(gt_item)


def ndcg(gt_item_index, pred_items_index):
	num_TP = len(gt_item_index)
	iDCG = 0
	if num_TP != 0:
		for i in range(num_TP):
			iDCG += 1/np.log2(i+1+1) # i从0开始，而DCG的计算从1开始
	DCG = 0
	k = len(pred_items_index)
	indices = torch.where(torch.isin(pred_items_index, gt_item_index))
	for i in range(len(indices)):
		DCG += 1/np.log2(indices[i]+1+1)
	nDCG = DCG/iDCG if iDCG !=0 else 0


def metrics(model, test_loader, top_k, device):
	HR, NDCG = [], []

	for user_item, label in test_loader:
		user = user_item[:, 0]
		item = user_item[:, 1]
		user = user.long().to(device=device)
		item = item.long().to(device=device)
        
		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k) # 得到predictions中前top_k大的值的索引
		recommends = torch.take(item, indices).cpu().numpy().tolist() # 从item里面根据indices索引取值，也就是预测的概率最高的top_k个MovieID

		# 找到label这个tensor中值等于1的索引
		indices_label_equal_1 = torch.nonzero(label).squeeze()
        
		HR.append(hit(indices_label_equal_1, indices))
        
		NDCG.append(ndcg(indices_label_equal_1, indices))

	return np.mean(HR[HR!=0]), np.mean(NDCG)