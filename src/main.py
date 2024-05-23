import os
import time
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
# from tensorboardX import SummaryWriter

import models
import config 
import util
import data_utils
import evaluate


parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", 
    type=int, 
    default=42, 
    help="Seed"
)
parser.add_argument(
    "--lr", 
    type=float, 
    default=0.001, 
    help="learning rate"              
)
parser.add_argument(
    "--dropout", 
    type=float,
    default=0.2,  
    help="dropout rate"
)
parser.add_argument(
    "--batch_size", 
    type=int, 
    default=256, 
    help="batch size for training"
)
parser.add_argument(
    "--epochs", 
    type=int,
    default=30,  
    help="training epoches"
)
parser.add_argument(
    "--top_k", 
    type=int, 
    default=10, 
    help="compute metrics@top_k"
)
parser.add_argument(
    "--factor_num", 
    type=int,
    default=8, 
    help="predictive factors numbers in the model"
)
parser.add_argument(
    "--layers",
    nargs='+', 
    default=[128, 64, 32, 16, 8],
    help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size."
)
parser.add_argument(
    "--num_ng", 
    type=int,
    default=4, 
    help="Number of negative samples for training set"
)
parser.add_argument(
    "--num_ng_test", 
    type=int,
    default=100, 
    help="Number of negative samples for test set"
)
parser.add_argument(
    "--out", 
    default=True,
    help="save model or not"
)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter()

util.seed_everything(args.seed)

ml_1m = pd.read_csv(
    config.DATA_PATH, 
    sep="::", 
    names = ['user_id', 'item_id', 'rating', 'timestamp'], 
    engine='python'
)

num_users = ml_1m['user_id'].nunique() + 1
num_items = ml_1m['item_id'].nunique() + 1

data = data_utils.NCF_Data(args, ml_1m)
train_loader = data.get_train_instance()
test_loader = data.get_test_instance()

if config.MODEL == "ml-1m_Neu_MF":
    model = models.NeuMF(args, num_users, num_items, num_layers = 3)
elif config.MODEL == "ml-1m_GMF":
    model = models.Generalized_Matrix_Factorization(args, num_users, num_items)
elif config.MODEL == "ml-1m_MLP":
    num_layers = config.MLP_LAYER
    args.factor_num = args.layers[len(args.layers) - num_layers - 1] // 2
    model = models.Multi_Layer_Perceptron(args, num_users, num_items, num_layers = num_layers)
model = model.to(device)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

best_hr = 0

result_HR10 = {config.MODEL: []}
result_NDCG10 = {config.MODEL: []}

for epoch in range(1, args.epochs + 1):
    model.train() # Enable dropout (if have).
    start_time = time.time()

    for user, item, label in train_loader:
        user = user.to(device)
        item = item.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        prediction = model(user, item)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
        # writer.add_scalar('loss/Train_loss', loss.item(), epoch)

    model.eval()
    HR, NDCG = evaluate.metrics(model, test_loader, args.top_k, device)
    result_HR10[config.MODEL].append(HR)
    result_NDCG10[config.MODEL].append(NDCG)
    print(f"{config.MODEL} - HR: {np.mean(HR):.3f}\tNDCG: {np.mean(NDCG):.3f}")
    
    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
            time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    

    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if args.out:
            if not os.path.exists(config.MODEL_PATH):
                os.mkdir(config.MODEL_PATH)
            torch.save(model, 
                '{}{}.pth'.format(config.MODEL_PATH, config.MODEL))
            
print(result_HR10)
print(result_NDCG10)

suffix = "" if config.MODEL != "ml-1m_MLP" else f"_{model.num_layers}Layer"

df_HR10 = pd.DataFrame(result_HR10)            
df_HR10.to_csv(f"../result/{config.MODEL}_HR10{suffix}.csv")

df_NDCG10 = pd.DataFrame(result_NDCG10)            
df_NDCG10.to_csv(f"../result/{config.MODEL}_NDCG10{suffix}.csv")

# writer.close()
print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
                                    best_epoch, best_hr, best_ndcg))