import torch
from engine import Engine
from utils import use_cuda
from torch import nn


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1) # 只输出一个值
        self.logistic = torch.nn.Sigmoid()  # 再用sigmoid回归一下

        # Initialize model parameters with a Gaussian distribution (with a mean of 0 and standard deviation of 0.01)
        if config['weight_init_gaussian']:  # 采用高斯分布初始化神经元权重， 3、
            for sm in self.modules():  # 递归遍历模型的所有模块
                if isinstance(sm, (nn.Embedding, nn.Linear)): # 判断sm是否是nn.Embedding或者nn.Linear的实例
                    print(sm)
                    torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)  # embedding之后做叉乘
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = GMF(config)  # 是gmf模型
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(GMFEngine, self).__init__(config)  # 是Engine的子类，继承了Engine的方法