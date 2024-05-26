import torch
from GMF import GMF
from MLP import MLP
from NeuMF import NCF
from load import metrics,load_dataset
from torch.utils.data import DataLoader
from dataset import NCFData
import os
import torch.backends.cudnn as cudnn

top_k = 10
print('load dataset')  
train_data, test_data, user_num, item_num, train_mat = load_dataset()
test_dataset = NCFData(test_data, item_num, train_mat, num_ng=0, is_training=False)  # 100
test_loader = DataLoader(test_dataset, batch_size=99 + 1, shuffle=False, num_workers=2)
print('finish_load')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU
cudnn.benchmark = True

for factor_num in [8,16,32,64]:
    model = GMF(int(user_num), int(item_num), factor_num)

    model.load_state_dict(torch.load(f'model_dict/GMF_dif_factor/GMF_model_{factor_num}.pth'))    
    model.to(device)    
    model.eval()  # 将模型设置为评估模式  
    HR, NDCG = metrics(model, test_loader, top_k, device)
    print(f'for GMF with factor: {factor_num}, model HR: {HR}, NDCG: {NDCG}')
    
for factor_num in [8,16,32,64]:
    model = MLP(int(user_num), int(item_num), factor_num, num_layers = 3, dropout = 0.0)
    
    model.load_state_dict(torch.load(f'model_dict/MLP_dif_factor/MLP_model_{factor_num}.pth'))
    model.to(device)    
    model.eval()  # 将模型设置为评估模式  
    HR, NDCG = metrics(model, test_loader, top_k, device)
    print(f'for MLP with factor: {factor_num}, model HR: {HR}, NDCG: {NDCG}')

for factor_num in [8,16,32,64]:    
    model = NCF(int(user_num), int(item_num), factor_num, num_layers = 3, dropout = 0.0)
    model.load_state_dict(torch.load(f'model_dict/NCF_dif_factor/NCF_model_{factor_num}.pth'))
    model.to(device)    
    model.eval()  # 将模型设置为评估模式  
    HR, NDCG = metrics(model, test_loader, top_k, device)
    print(f'for NCF with factor: {factor_num}, model HR: {HR}, NDCG: {NDCG}')


