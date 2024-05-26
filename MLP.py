import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


class MLP(nn.Module):# []
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
        self.num_layers = num_layers
        super(MLP, self).__init__()
        bottom = factor_num * (2 ** num_layers)
        if(num_layers != 0):
            self.embed_user_MLP = nn.Embedding(user_num, int(bottom/2))
            self.embed_item_MLP = nn.Embedding(item_num, int(bottom/2))
            MLP_modules = []
            for i in range(num_layers):
                input_size = int(bottom/(2**i))
                MLP_modules.append(nn.Dropout(p=dropout))
                MLP_modules.append(nn.Linear(input_size, int(input_size / 2)))
                MLP_modules.append(nn.ReLU())
            self.MLP_layers = nn.Sequential(*MLP_modules)
            self.predict_layer = nn.Linear(factor_num, 1)
                       
        else:
            self.embed_user_MLP = nn.Embedding(user_num, factor_num)
            self.embed_item_MLP = nn.Embedding(item_num, factor_num)
             
            self.predict_layer = nn.Linear(2*factor_num, 1)
        self.sigmoid = nn.Sigmoid()
 
        self._init_weight_()
 
    def _init_weight_(self):
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        
        if(self.num_layers != 0):
            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
 
    def forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        if(self.num_layers != 0):
            output_MLP = self.MLP_layers(interaction)
            prediction = self.predict_layer(output_MLP)
        else:
            prediction = self.predict_layer(interaction)            
        logit = self.sigmoid(prediction)
        return logit.view(-1)    