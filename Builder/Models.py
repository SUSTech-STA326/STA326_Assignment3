import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


class GMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(GMF, self).__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=num_factors).cuda()  
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=num_factors).cuda()  
        self.apply(self.normalize)
        self.device = torch.device('cuda')

    def forward(self, user_ids, item_ids):
        user_ids = user_ids.to(self.device)  
        item_ids = item_ids.to(self.device)  
        user_embedding = self.user_emb(user_ids)
        item_embedding = self.item_emb(item_ids)
        output = (user_embedding * item_embedding).sum(1)
        return F.relu(output)  # 修改激活函数为ReLU

    def normalize(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight.data, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight.data, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0)
                
class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers):
        super(MLP, self).__init__()
        self.user_emb = nn.Embedding(num_users, layers[0] // 2 if layers else 8).cuda()  # 移到CUDA设备上
        self.item_emb = nn.Embedding(num_items, layers[0] // 2 if layers else 8).cuda()  # 移到CUDA设备上
        self.fc_layers = nn.ModuleList()
        self.device = torch.device('cuda')
        for idx in range(1, len(layers)):
            self.fc_layers.append(nn.Linear(layers[idx - 1], layers[idx]).cuda())  # 移到CUDA设备上
        self.output_layer = nn.Linear(layers[-1] if layers else 16, 1).cuda()  # 移到CUDA设备上

    def forward(self, user_ids, item_ids):
        user_ids = user_ids.to(self.device)  # 移到CUDA设备上
        item_ids = item_ids.to(self.device)  # 移到CUDA设备上
        user_embedding = self.user_emb(user_ids)
        item_embedding = self.item_emb(item_ids)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for layer in self.fc_layers:
            vector = torch.relu(layer(vector))
        output = self.output_layer(vector)
        return output.sigmoid()

def MLP_with_hidden_layers(num_users, num_items, num_factors, num_hidden_layers):
    if num_hidden_layers == 0:
        layers = []  # 没有隐藏层
    else:
        layers = [num_factors] * num_hidden_layers  # 为简单起见，与潜在因素大小相同
    return MLP(num_users, num_items, layers)


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors, mlp_layers):
        super(NeuMF, self).__init__()
        self.gmf = GMF(num_users, num_items, num_factors).cuda()  
        self.mlp = MLP_with_hidden_layers(num_users, num_items, num_factors, 3).cuda()  # 修改隐藏层层数
        self.output_layer = nn.Linear(2, 1).cuda()  
        self.device = torch.device('cuda')

    def forward(self, user_ids, item_ids):
        user_ids = user_ids.to(self.device)  
        item_ids = item_ids.to(self.device)  
        gmf_output = self.gmf(user_ids, item_ids)
        mlp_output = self.mlp(user_ids, item_ids)
        gmf_output = gmf_output.unsqueeze(1)
        concatenated_output = torch.cat((gmf_output, mlp_output), dim=-1)
        output = self.output_layer(concatenated_output)
        return torch.sigmoid(output)  # 修改激活函数为sigmoid
