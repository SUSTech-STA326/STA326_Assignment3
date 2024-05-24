import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self._num_users = config['num_users']
        self._num_items = config['num_items']
        self._X = config['layer_X']
        self._factor = config['factor']
        self._embedding_size_gmf = self._factor
        self._embedding_size_mlp = int(self._factor*(2**(self._X-1)))

        self._embedding__user_gmf = nn.Embedding(self._num_users, self._embedding_size_gmf)
        self._embedding__item_gmf = nn.Embedding(self._num_items, self._embedding_size_gmf)

        self._embedding__user_mlp = nn.Embedding(self._num_users, self._embedding_size_mlp)
        self._embedding__item_mlp = nn.Embedding(self._num_items, self._embedding_size_mlp)

        if self._X > 0:
            self._fc_layers = nn.ModuleList()
            for idx in range(self._X-1, -1, -1):
                in_size = self._factor*(2**(idx+1))
                out_size = self._factor*(2**idx)
                self._fc_layers.append(nn.Linear(in_size, out_size))
        self._out_fc = nn.Linear(self._factor, 1, bias=False)
        
        self._activate1 = nn.Sigmoid()
        self._activate2 = nn.ReLU()
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self._embedding__user_gmf.weight, std=0.01)
        nn.init.normal_(self._embedding__user_mlp.weight, std=0.01)
        nn.init.normal_(self._embedding__item_gmf.weight, std=0.01)
        nn.init.normal_(self._embedding__item_mlp.weight, std=0.01)

        if self._X > 0:
            for m in self._fc_layers:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity='relu')
        nn.init.xavier_uniform_(self._out_fc.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def __repr__(self):
        return ""
    
class GMF(NCF):
    def __init__(self, config):
        NCF.__init__(self, config)

    def forward(self, user_idx, item_idx):
        user_embedding = self._embedding__user_gmf(user_idx)
        item_embedding = self._embedding__item_gmf(item_idx)
        pointwise_vector = torch.mul(user_embedding, item_embedding)
        logit = self._out_fc(pointwise_vector)
        prob = self._activate1(logit)
        return prob.squeeze(1)
    
class MLP(NCF):
    def __init__(self, config):
        NCF.__init__(self, config)

    def forward(self, user_idx, item_idx):
        user_embedding = self._embedding__user_mlp(user_idx)
        item_embedding = self._embedding__item_mlp(item_idx)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        if self._X > 0:
            for _, layer in enumerate(self._fc_layers):
                vector = layer(vector)
                vector = self._activate2(vector)
        logit = self._out_fc(vector)
        prob = self._activate1(logit)
        return prob.squeeze(1)
    
class NeuMF(NCF):
    def __init__(self, config):
        NCF.__init__(self, config)
        self._neumf_fc = nn.Linear(self._factor*2, 1, bias=False)
        nn.init.xavier_normal_(self._neumf_fc.weight)
    
    def forward(self, user_idx, item_idx):
        user_embedding_gmf = self._embedding__user_gmf(user_idx)
        item_embedding_gmf = self._embedding__item_gmf(item_idx)
        pointwise_vector_gmf = torch.mul(user_embedding_gmf, item_embedding_gmf)

        user_embedding_mlp = self._embedding__user_mlp(user_idx)
        item_embedding_mlp = self._embedding__item_mlp(item_idx)
        vector_mlp = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)

        if self._X > 0:
            for _, layer in enumerate(self._fc_layers):
                vector_mlp = layer(vector_mlp)
                vector_mlp = self._activate2(vector_mlp)

        vector_neumf = torch.cat([pointwise_vector_gmf, vector_mlp], dim=-1)
        logit = self._neumf_fc(vector_neumf)
        prob = self._activate1(logit)
        return prob.squeeze(1)