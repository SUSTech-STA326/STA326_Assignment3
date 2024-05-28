import torch.nn as nn
import torch 
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)

class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc = nn.Linear(embedding_size, 1)
        
    def forward(self, user_ids, item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        element_product = user_embedding * item_embedding
        logits = self.fc(element_product)
        return logits.view(-1)


class MLP(nn.Module):
    def __init__(self, num_users, num_items, hidden_sizes):
        super(MLP, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        layers = []
        input_size = num_users + num_items
        layer_sizes = [input_size] + hidden_sizes + [1] 
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())  
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_onehot = F.one_hot(user_ids, num_classes=self.num_users).float()
        item_onehot = F.one_hot(item_ids, num_classes=self.num_items).float()
        input_vec = torch.cat([user_onehot, item_onehot], dim=1)
        return self.mlp(input_vec)

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=10, mlp_layers=[64, 32], dropout=0.2):
        super(NeuMF, self).__init__()
        self.user_embedding_mf = nn.Embedding(num_users, mf_dim)
        self.item_embedding_mf = nn.Embedding(num_items, mf_dim)
        layers = []
        input_size = mf_dim * 2
        layer_sizes = [input_size] + mlp_layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(mlp_layers[-1] + mf_dim, 1)

    def forward(self, user_ids, item_ids):
        user_embedding_mf = self.user_embedding_mf(user_ids)
        item_embedding_mf = self.item_embedding_mf(item_ids)
        mf_output = torch.mul(user_embedding_mf, item_embedding_mf)  # Element-wise multiplication

        user_embedding_mlp = self.user_embedding_mf(user_ids)
        item_embedding_mlp = self.item_embedding_mf(item_ids)
        mlp_input = torch.cat((user_embedding_mlp, item_embedding_mlp), dim=1)
        mlp_output = self.mlp(mlp_input)

        concat_output = torch.cat((mf_output, mlp_output), dim=1)
        output = self.output_layer(concat_output).squeeze()
        return output