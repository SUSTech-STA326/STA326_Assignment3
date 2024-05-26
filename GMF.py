import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


class GMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(GMF, self).__init__()
 
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.predict_layer = nn.Linear(factor_num, 1)
        self.sigmoid = nn.Sigmoid()
 
        self._init_weight_()
 
    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
 
    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        prediction = self.predict_layer(output_GMF)
        logits = self.sigmoid(prediction)
        return logits.view(-1)