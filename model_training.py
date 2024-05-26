import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  # 如果你使用 tqdm 来显示进度条
import os  # 如果你操作文件或目录

# 你的自定义模块（确保路径正确或已包含在 PYTHONPATH 中）
from MLP import MLP  # 你提到的但未给出代码
from GMF import GMF  # 你可能写错了，你提到了 GLM 而不是 GMF
from NeuMF import NCF  # 确保这是正确的导入路径
from dataset import NCFData
from load import load_dataset,hit,ndcg,metrics
import torch.backends.cudnn as cudnn




dropout=0.0
lr = 0.001
epoch_num = 20
top_k = 10
factor_num = 8
num_layers = 3
hr1 = []
ndcg1 = []

print('load dataset')  
train_data, test_data, user_num, item_num, train_mat = load_dataset()

train_dataset = NCFData(train_data, item_num, train_mat, num_ng=4, is_training=True)  # neg_items=4,default
test_dataset = NCFData(test_data, item_num, train_mat, num_ng=0, is_training=False)  # 100
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
# every user have 99 negative items and one positive items，so batch_size=100
test_loader = DataLoader(test_dataset, batch_size=99 + 1, shuffle=False, num_workers=2)
print('finish_load')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU
cudnn.benchmark = True

print('Train GMF model')

for factor_num in [8, 16, 32, 64]:
    model = GMF(int(user_num), int(item_num), factor_num)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCELoss()

    best_hr, best_ndcg = 0.0, 0.0

    for epoch in tqdm(range(epoch_num), desc="Training Progress"):
        model.train()
        train_loader.dataset.ng_sample()
        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

        model.eval()
        HR, NDCG = metrics(model, test_loader, top_k, device)
        if HR > best_hr or NDCG > best_ndcg:
            best_hr, best_ndcg = HR, NDCG
            # 仅在性能提升时保存模型
            torch.save(model.state_dict(), f'model_dict/GMF_dif_factor/GMF_model_{factor_num}.pth')
    print(f"best HR: {best_hr} best NDCG: {best_ndcg} for GLM model")    
    
print('Train MLP model num_layers = 3')   

for factor_num in [8, 16, 32, 64]:
    
    model = MLP(int(user_num), int(item_num), factor_num, num_layers, dropout)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCELoss()

    best_hr, best_ndcg = 0.0, 0.0

    for epoch in tqdm(range(epoch_num), desc="Training Progress"):
        model.train()
        train_loader.dataset.ng_sample()
        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

        model.eval()
        HR, NDCG = metrics(model, test_loader, top_k, device)
        if HR > best_hr or NDCG > best_ndcg:
            best_hr, best_ndcg = HR, NDCG
            # 仅在性能提升时保存模型
            torch.save(model.state_dict(), f'diff_layer_dict/MLP_layer_{num_layers}_factor_{factor_num}.pth')

    print(f"num of factor: {factor_num}\nnum of layer: {num_layers}\nbest hr: {best_hr}\nbest_ndcg: {best_ndcg}")

        
print('Train NCF model')
    
for factor_num in [8, 16, 32, 64]:
    model = NCF(int(user_num), int(item_num), factor_num, num_layers, dropout=0.0)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCELoss()

    best_hr, best_ndcg = 0.0, 0.0

    for epoch in tqdm(range(epoch_num), desc="Training Progress"):
        model.train()
        train_loader.dataset.ng_sample()
        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

        model.eval()
        HR, NDCG = metrics(model, test_loader, top_k, device)
        if HR > best_hr or NDCG > best_ndcg:
            best_hr, best_ndcg = HR, NDCG
            # 仅在性能提升时保存模型
            torch.save(model.state_dict(), f'model_dict/NCF_dif_factor/NCF_model_{factor_num}.pth')
    print(f"best HR: {best_hr} best NDCG: {best_ndcg} for MLP model")
    
print('all model have trained and saved in model_dict director')