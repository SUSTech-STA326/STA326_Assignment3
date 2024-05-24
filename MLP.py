import torch
from GMF import GMF
from engine import Engine
from utils import resume_checkpoint
from torch import nn


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        #全连接层,设定每个层的大小
        self.fc_layers = torch.nn.ModuleList()
        for _, (in_dim, out_dim) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_dim, out_dim))
        #全连接最后一层 
        self.affine_layer = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        #用正态分布给embedding和linear layer初始化权重
        for module in self.modules():
                if isinstance(module, (nn.Embedding, nn.Linear)):
                    torch.nn.init.normal_(module.weight.data, 0.0, 0.01)

    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        #contact user和item两个向量
        c_vector = torch.cat([user_embedding, item_embedding], dim=-1)  
        for i, _ in enumerate(range(len(self.fc_layers))):
            c_vector = self.fc_layers[i](c_vector)
            #使用relu作为激活函数
            c_vector = torch.nn.ReLU()(c_vector)
        logit = self.affine_layer(c_vector)
        score = self.logistic(logit)
        return score

    def init_weight(self):
        pass



class MLPEngine(Engine):
    def __init__(self, config):
        self.model = MLP(config)       
        self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)

