import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []

	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)


# def hr_ndcg_at_k(model, test_loader, K=10):
#     HR = []
#     NDCG = []
#     for user, item, label in test_loader:
#         user = user.cuda()
#         item = item.cuda()
#         predictions = model(user, item)
#         _, indices = torch.topk(predictions, K)
#         recommends = torch.take(item, indices).cpu().numpy().tolist()

#         positive_items = item[label == 1]
#         if positive_items.nelement() == 0:
#             # Handle case where there are no positive items
#             HR.append(0)
#             NDCG.append(0)
#         else:
#             gt_item = positive_items.item()  # Assumption: each user has one positive item

#             # Calculate HR
#             if gt_item in recommends:
#                 HR.append(1)
#             else:
#                 HR.append(0)

#             # Calculate NDCG
#             if gt_item in recommends:
#                 index = recommends.index(gt_item)
#                 NDCG.append(np.reciprocal(np.log2(index + 2)))
#             else:
#                 NDCG.append(0)

#     return np.mean(HR), np.mean(NDCG)