import torch
from gmf import GMF
from mlp import MLP
from engine import Engine
from utils import use_cuda, resume_checkpoint
from torch import nn


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.gmf = GMF(config)
        self.mlp = MLP(config)
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim'], out_features=1)
        self.logistic = torch.nn.Sigmoid()
        if config['weight_init_gaussian']:
            torch.nn.init.normal_(self.affine_output.weight.data, 0.0, 0.01)
        
    def forward(self, user_indices, item_indices):
        mlp_vector = self.mlp(user_indices, item_indices)
        mf_vector = self.gmf(user_indices, item_indices)
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

class NeuMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = NeuMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(NeuMFEngine, self).__init__(config)
        print(self.model)