import torch
import torch.nn as nn
import torch.nn.functional as F 

class NCF_MLP_0(nn.Module):
    def __init__(self, user_num, item_num, factor_num, dropout):
        super(NCF_MLP_0, self).__init__()

        self.dropout = dropout

        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        self.predict_layer = nn.Linear(factor_num * 2, 1)  # Directly project concatenated embeddings to prediction

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        if self.predict_layer.bias is not None:
            self.predict_layer.bias.data.zero_()

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)

        concat = torch.cat((embed_user, embed_item), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)
