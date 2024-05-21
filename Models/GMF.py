import torch
from torch import nn


import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def init_weights(m):
    if isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)

class GMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, reg=[0, 0]):
        super(GMF, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
        self.linear = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = input.long()
        MF_Embedding_User = self.MF_Embedding_User(input[:, 0])
        MF_Embedding_Item = self.MF_Embedding_Item(input[:, 1])

        predict = torch.mul(MF_Embedding_User, MF_Embedding_Item)

        linear = self.linear(predict)
        output = self.sigmoid(linear)
        output = output.squeeze(-1)
        return output

class GMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(GMF, self).__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=num_factors)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=num_factors)
        self.apply(init_weights)  # Apply custom initialization
        if isinstance(self, nn.Embedding):
            torch.nn.init.normal_(self.weight.data, mean=0.0, std=0.01)
        elif isinstance(self, nn.Linear):
            torch.nn.init.normal_(self.weight.data, mean=0.0, std=0.01)
            if self.bias is not None:
                torch.nn.init.constant_(self.bias.data, 0)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_emb(user_ids)
        item_embedding = self.item_emb(item_ids)
        output = (user_embedding * item_embedding).sum(1)
        return output.sigmoid()