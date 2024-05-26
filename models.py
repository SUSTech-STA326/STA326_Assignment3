import torch
from torch import nn

class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim_mlp']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

        if config['weight_init_gaussian']:
            for sm in self.modules():
                if isinstance(sm, (nn.Embedding, nn.Linear)):
                    torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)
        
    def forwardX(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = torch.nn.ReLU()(x)
        return x

    def forward(self, user_indices, item_indices):
        x = self.forwardX(user_indices, item_indices)
        logits = self.affine_output(x)
        rating = self.logistic(logits)
        return rating
    
class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim_mf']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_layer = nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

        if config['weight_init_gaussian']:
            for sm in self.modules():
                if isinstance(sm, (nn.Embedding, nn.Linear)):
                    torch.nn.init.normal_(sm.weight.data, 0.0, 0.01)
        
    def forwardX(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        x = torch.mul(user_embedding, item_embedding)
        return x

    def forward(self, user_indices, item_indices):
        x = self.forwardX(user_indices, item_indices)
        logits = self.affine_layer(x)
        rating = self.logistic(logits)
        return rating
    
class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.mlp = MLP(config)
        self.gmf = GMF(config)
        self.affine_layer = nn.Linear(in_features=self.config['layers'][-1] + self.config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        x_mlp = self.mlp.forwardX(user_indices, item_indices)
        x_gmf = self.gmf.forwardX(user_indices, item_indices)
        x = torch.cat([x_mlp, x_gmf], dim=-1)
        logits = self.affine_layer(x)
        rating = self.logistic(logits)
        return rating