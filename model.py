import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num: int, num_layers = 3, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(user_num, int(factor_num * (2 ** (num_layers - 1))))
        self.embed_item_MLP = nn.Embedding(item_num, int(factor_num * (2 ** (num_layers - 1))))

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            # MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
    
    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        self.output_GMF = embed_user_GMF * embed_item_GMF
        
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        self.output_MLP = self.MLP_layers(interaction)
        pass
    
class GMF(NCF):
    def __init__(self, user_num, item_num, factor_num, num_layers=3, GMF_model=None, MLP_model=None):
        super(GMF, self).__init__(user_num, item_num, factor_num, num_layers, GMF_model, MLP_model)
        self.predict_layer = nn.Linear(factor_num, 1, bias=False)
        
        self._init_weight_()
    
    def _init_weight_(self):
        super(GMF, self)._init_weight_()
        
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, user, item):
        super(GMF, self).forward(user, item)
        prediction = torch.sigmoid(self.predict_layer(self.output_GMF))
        return prediction
    
class MLP(NCF):
    def __init__(self, user_num, item_num, factor_num, num_layers=3, GMF_model=None, MLP_model=None):
        super(MLP, self).__init__(user_num, item_num, factor_num, num_layers, GMF_model, MLP_model)
        self.predict_layer = nn.Linear(factor_num, 1, bias= False)
        
        self._init_weight_()
    
    def _init_weight_(self):
        super(MLP, self)._init_weight_()
        
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, user, item):
        super(MLP, self).forward(user, item)
        prediction = torch.sigmoid(self.predict_layer(self.output_MLP))
        return prediction
    
class NeuMF(NCF):
    def __init__(self, user_num, item_num, factor_num, num_layers=3, GMF_model=None, MLP_model=None, preTrain = True):
        super(NeuMF, self).__init__(user_num, item_num, factor_num, num_layers, GMF_model, MLP_model)
        self.predict_layer = nn.Linear(factor_num * 2, 1, bias= False)
        self.preTrain = preTrain
        
        self._init_weight_()
    
    def _init_weight_(self):
        if not self.preTrain:
            super(NeuMF, self)._init_weight_()
            
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
            
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            for (m1, m2) in zip(
                self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
                    
            predict_weight = torch.cat([self.GMF_model.predict_layer.weight, self.MLP_model.predict_layer.weight], dim=1)
            # precit_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            # self.predict_layer.bias.data.copy_(0.5 * precit_bias)
    
    def forward(self, user, item):
        super(NeuMF, self).forward(user, item)
        output_NeuMF = torch.cat((self.output_GMF, self.output_MLP), -1)
        prediction = torch.sigmoid(self.predict_layer(output_NeuMF))
        return prediction
        
        
        
        
