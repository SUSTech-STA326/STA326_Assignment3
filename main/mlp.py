import torch
from gmf import GMF
from engine import Engine
from utils import use_cuda, resume_checkpoint
from torch import nn

class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # Initialize model parameters with a Gaussian distribution (with a mean of 0 and standard deviation of 0.01)
        if config['weight_init_gaussian']:
            for sm in self.modules():
                if isinstance(sm, (nn.Embedding, nn.Linear)):
                    torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
        return vector

class MLPWithOutput(nn.Module):
    def __init__(self, config):
        super(MLPWithOutput, self).__init__()
        self.mlp = MLP(config)
        self.affine_output = nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = nn.Sigmoid()

        # Initialize Linear layer with a Gaussian distribution
        if config['weight_init_gaussian']:
            torch.nn.init.normal_(self.affine_output.weight.data, 0.0, 0.01)

    def forward(self, user_indices, item_indices):
        vector = self.mlp(user_indices, item_indices)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating


class MLPEngine(Engine):
    """Engine for training & evaluating MLP model"""
    def __init__(self, config):
        self.model = MLPWithOutput(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)
