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

        # Initialize embedding layers with a Gaussian distribution
        if config['weight_init_gaussian']:
            torch.nn.init.normal_(self.embedding_user.weight.data, 0.0, 0.01)
            torch.nn.init.normal_(self.embedding_item.weight.data, 0.0, 0.01)

    def forward(self, user_indices, item_indices):
        element_product = torch.mul(self.embedding_user(user_indices), self.embedding_item(item_indices))
        return element_product

class GMFWithOutput(nn.Module):
    def __init__(self, config):
        super(GMFWithOutput, self).__init__()
        self.gmf = GMF(config)
        self.affine_output = nn.Linear(in_features=config['latent_dim'], out_features=1)
        self.logistic = nn.Sigmoid()

        # Initialize Linear layer with a Gaussian distribution
        if config['weight_init_gaussian']:
            torch.nn.init.normal_(self.affine_output.weight.data, 0.0, 0.01)

    def forward(self, user_indices, item_indices):
        element_product = self.gmf(user_indices, item_indices)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = GMFWithOutput(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(GMFEngine, self).__init__(config)
        print(self.model)

