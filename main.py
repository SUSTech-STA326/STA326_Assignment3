# from data_pre import Dataset
from model import NeuralCollaborativeFiltering
import tqdm
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from datasets.movielens import MovieLens1MDataset, MovieLensDataset_val

def get_dataset(name, path, mode = None):
    if name == 'movielens1M':
        if mode == "val":
            print("val mode")
            return MovieLensDataset_val(path)
        return MovieLens1MDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    losses = []
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y, _ = model(fields)
        loss = criterion(y, target.float())
        losses.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            losses.append(total_loss / log_interval)
            total_loss = 0
    return losses

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y, _ = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)

def test_metric(model, data_loader, device, k=10):
    model.eval()
    target, predict, indexes, pos_indexes= list(), list(), list(), list()
    HR, NDCG = [],[]
    with torch.no_grad():
        i = 0
        for fields, label in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, label = fields.to(device), label.to(device)
            y, index = model(fields)
            target.extend(label.tolist())
            predict.extend(y.tolist())
            indexes.extend(index[: ,0].tolist())
            pos_indexes.extend(index[: ,1].tolist())

        combined = pd.DataFrame(zip(indexes, pos_indexes, predict, target), columns=['index', 'pos_index', 'predict', 'target'])
        assert len(target) == len(predict) == combined.shape[0] == len(indexes) == len(pos_indexes)
        groups = combined.groupby('index')
        for name, group in groups:
            HR.append(HR_at_k(group, k))
            NDCG.append(NDCG_at_k(group, k))
    return np.mean(HR), np.mean(NDCG)

def HR_at_k(df, k):
    df = df.sort_values(by='predict', ascending=False)
    # 查找前k个里有没有target_index == 1
    for i in range(k):
        if df.iloc[i]['target'] == 1:
            return 1
    return 0

def NDCG_at_k(df, k):
    df = df.sort_values(by='predict', ascending=False)
    dcg = 0
    for i in range(k):
        if df.iloc[i]['target'] == 1:
            dcg += 1 / np.log2(i + 2)
    return dcg 

def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         hidden):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    test_length = len(dataset) - train_length
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, test_length))
    valid_dataset = get_dataset(dataset_name, dataset_path,mode="val")

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    field_dims = dataset.field_dims
    print("hidden states: ", hidden)
    assert isinstance(dataset, MovieLens1MDataset)
    model_NCF = NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=hidden[0], dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx,
                                            mode='NCF')
    model_MLP = NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=hidden[0], dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx,
                                            mode='MLP')
    model_GMF = NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=hidden[0], dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx,
                                            mode='GMF')
    

    criterion = torch.nn.BCELoss()  # l = -y * log(x) - (1-y) * log(1-x)
    optimizer_NCF = torch.optim.Adam(params=model_NCF.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_MLP = torch.optim.Adam(params=model_MLP.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_GMF = torch.optim.Adam(params=model_GMF.parameters(), lr=learning_rate, weight_decay=weight_decay)

    HRs, NDCGs, LOSSes = {}, {}, {}
    HRs["NCF"] = []
    HRs["MLP"] = []
    HRs["GMF"] = []
    NDCGs["NCF"] = []
    NDCGs["MLP"] = []
    NDCGs["GMF"] = []
    LOSSes["NCF"] = []
    LOSSes["MLP"] = []
    LOSSes["GMF"] = []

    for epoch_i in range(epoch):
        losses_NCF = train(model_NCF, optimizer_NCF, train_data_loader, criterion, device)
        losses_MLP = train(model_MLP, optimizer_MLP, train_data_loader, criterion, device)
        losses_GMF = train(model_GMF, optimizer_GMF, train_data_loader, criterion, device)
        HR_NCF, NDCG_NCF = test_metric(model_NCF, valid_data_loader, device, k=10)
        HR_MLP, NDCG_MLP = test_metric(model_MLP, valid_data_loader, device, k=10)
        HR_GMF, NDCG_GMF = test_metric(model_GMF, valid_data_loader, device, k=10)
        HRs["NCF"].append(HR_NCF)
        HRs["MLP"].append(HR_MLP)
        HRs["GMF"].append(HR_GMF)
        NDCGs["NCF"].append(NDCG_NCF)
        NDCGs["MLP"].append(NDCG_MLP)
        NDCGs["GMF"].append(NDCG_GMF)
        LOSSes["NCF"]+=losses_NCF
        LOSSes["MLP"]+=losses_MLP
        LOSSes["GMF"]+=losses_GMF
        print(f"epoch: {epoch_i}, NCF HR: {HR_NCF}, NCF NDCG: {NDCG_NCF}")
        print(f"epoch: {epoch_i}, MLP HR: {HR_MLP}, MLP NDCG: {NDCG_MLP}")
        print(f"epoch: {epoch_i}, GMF HR: {HR_GMF}, GMF NDCG: {NDCG_GMF}")

    log = {}
    log['HRs'] = HRs
    log['NDCGs'] = NDCGs
    # save the metrics as plk
    with open(f'save/factor_{hidden[1]}_logs.pkl', 'wb') as f:
        pickle.dump(HRs, f)

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', default='Data/')
    parser.add_argument('--model_name', default='afi')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    hiddens = [[(32,16,8), 8], [(32,16,16),16], [(64,32,32),32], [(128,64,64),64]]
    for hidden in hiddens:
        main('movielens1M',
            'Data/',
            args.model_name,
            args.epoch,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.device,
            args.save_dir,
            hidden)