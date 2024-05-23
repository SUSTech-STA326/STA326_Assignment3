from torch.utils.data import Dataset, DataLoader
import torch
import math
from dataset import loadTest

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
test = loadTest("./Data/ml-1m")
testRatings, testNegatives = test.testRatings, test.testNegatives

def evaluate(model,topk):
    class testDataset(Dataset):
        def __init__(self, rating, negative_lists):
            self.rating = rating
            self.negative_lists = negative_lists
        def __len__(self):
            return len(self.rating)
        def __getitem__(self, index):
            return self.rating[index], self.negative_lists[index]

    def HR_NDCG(testloader):
        model.eval()
        ht = 0; ndcg = 0
        leng = 0
        with torch.no_grad():
            for rating, negatives in testloader:
                user_idxs = rating[0].clone().detach().to(device)
                pos_item_idxs = rating[1].clone().detach().to(device)
                neg_item_idxs = torch.stack(negatives).to(device) # 99*256

                pos_scores = model(user_idxs, pos_item_idxs).unsqueeze(1) # (batch_size, 1)
                neg_scores = torch.tensor([model(user_idxs,items).tolist() for items in neg_item_idxs]).t().to(device)  # (batch_size, num_negatives)
                all_scores = torch.cat((pos_scores, neg_scores), dim=1)  # (batch_size, num_negatives+1)

                # calculate HR
                _, topk_indices = torch.topk(all_scores, topk, dim=1, largest=True, sorted=True)
                ht += torch.sum((topk_indices == 0).int()).item()  # 0 is the index of positive example in concatenated scores

                # calculate NDCG
                sorted_scores, _ = torch.sort(all_scores, dim=1, descending=True)
                rankings = torch.argmax((sorted_scores == pos_scores.expand_as(sorted_scores)).int(), dim=1)

                ndcg += sum([math.log(2)/math.log(x+2) if x < 10 else 0 for x in rankings.tolist()])

        return ht / len(testloader.dataset), ndcg / len(testloader.dataset)

    test_dataset = testDataset(testRatings, testNegatives)
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=2)
    hr, ndcg = HR_NDCG(test_loader)
    return hr, ndcg