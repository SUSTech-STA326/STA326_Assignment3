import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from Dataset import Data, Data2dataset
from model import GMF, MLP, NeuMF
from evaluate import metrics
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GMF')
parser.add_argument('--lr', type= float, default=0.001)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--topk', type= int, default=10)
parser.add_argument('--num_ng', type= int, default=4)
parser.add_argument('--factor_num', type= int, default=8)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--out',type= bool, default=True)
parser.add_argument('--epochs', type= int, default=30)

args = parser.parse_args()

cudnn.benchmark = True

data_path = 'Data/ml-1m'
model_path = './res/model/'
GMF_model_path = './res/model/GMF.pth'
MLP_model_path = './res/model/MLP-3.pth'
resPath = './res/plot/'

# train function definition
def train(model, optimizer, loss_function, train_loader, test_loader, epochs, topk):
    best_hr = 0
    best_ndcg = 0
    hr_epochs = []
    ndcg_epochs = []
    loss_epochs = []
    for epoch in range(epochs):
        loss_epoch = []
        model.train()
        train_loader.dataset.ng_sample()
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for user, item, label in train_loader_tqdm:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()
            
            model.zero_grad()
            prediction = model(user, item).view(-1)
            loss = loss_function(prediction, label)
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            
            train_loader_tqdm.set_postfix(loss=loss.item())
        
        avg_loss = sum(loss_epoch)/len(loss_epoch)
        model.eval()
        hr, ndcg = metrics(model, test_loader, topk)
        hr_epochs.append(hr)
        ndcg_epochs.append(ndcg)
        loss_epochs.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, HR: {hr:.4f}, NDCG: {ndcg:.4f}, Loss: {avg_loss:.4f}")
        
        if hr > best_hr:
            best_hr = hr
            if args.out:
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(model,'{}{}.pth'.format(model_path, args.model))
        
        if ndcg > best_ndcg:
            best_ndcg = ndcg
                
    print("End. Best: HR = {:.3f}, NDCG = {:.3f}".format( best_hr, best_ndcg))
    return hr_epochs, ndcg_epochs, loss_epochs

def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
# Data Prepare
data = Data(data_path)
user_num = data.num_user
item_num = data.num_item

train_dataset = Data2dataset(data = data, num_ng= args.num_ng, istrain= True)
test_dataset = Data2dataset(data = data, num_ng= 0, istrain= False)
train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset,batch_size=100, shuffle=False, num_workers=0)

# Model Set
seed_everything(42)

if args.model == 'NeuMF-pre':
    assert os.path.exists(GMF_model_path), 'lack of GMF model'
    assert os.path.exists(MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(GMF_model_path)
    MLP_model = torch.load(MLP_model_path)
    
    model = NeuMF(user_num, item_num, args.factor_num, args.num_layers, GMF_model, MLP_model, preTrain = True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.model == 'NeuMF':
    model = NeuMF(user_num, item_num, args.factor_num, args.num_layers,  preTrain = False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
elif args.model == 'GMF':
    model = GMF(user_num, item_num, args.factor_num)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
else:
    model = MLP(user_num, item_num, args.factor_num, args.num_layers)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

model.cuda()
loss_function = nn.BCELoss()

hr, ndcg, loss = train(model, optimizer, loss_function, train_loader, test_loader, args.epochs, args.topk)

if not os.path.exists(resPath):
    os.mkdir(resPath)
with open('{}{}_res.txt'.format(resPath, args.model), 'w') as f:
    f.write('HR: {}\n'.format(hr))
    f.write('NDCG: {}\n'.format(ndcg))
    f.write('Loss: {}\n'.format(loss))



# Training
# best_hr = 0
# epochs = 30
# for epoch in range(epochs):
#     model.train()
#     train_loader.dataset.ng_sample()
    
#     train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

#     for user, item, label in train_loader_tqdm:
#         user = user.cuda()
#         item = item.cuda()
#         label = label.float().cuda()

#         model.zero_grad()
#         prediction = model(user, item).view(-1)
#         loss = loss_function(prediction, label)
#         loss.backward()
#         optimizer.step()

#         train_loader_tqdm.set_postfix(loss=loss.item())

#     model.eval()
#     hr, ndcg = metrics(model, test_loader, 10)
#     print(f"Epoch {epoch+1}/{epochs}, HR: {hr:.4f}, NDCG: {ndcg:.4f}")

#     if hr > best_hr:
#         best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
#         if args.out:
#             if not os.path.exists(model_path):
#                 os.mkdir(model_path)
#             torch.save(model,'{}{}.pth'.format(model_path, args.model))
            
# print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))
  