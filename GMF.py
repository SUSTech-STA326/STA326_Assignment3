import torch
from engine import Engine
from torch import nn


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)       
        self.affine_layer=torch.nn.Linear(self.latent_dim,1)
        self.logistic = torch.nn.Sigmoid()
        #用正态分布给embedding和linear layer初始化权重
        for module in self.modules():
                if isinstance(module, (nn.Embedding, nn.Linear)):
                    torch.nn.init.normal_(module.weight.data, 0.0, 0.01)


    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        element_product = torch.mul(user_embedding, item_embedding)
        logit = self.affine_layer(element_product)
        score = self.logistic(logit)
        return score


class GMFEngine(Engine):
    def __init__(self, config):
        self.model = GMF(config)
        self.model.cuda()
        super(GMFEngine, self).__init__(config)
