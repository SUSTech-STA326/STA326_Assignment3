import copy
import torch
import torch.nn as nn

class Base_Module(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Base_Module, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = args.factor_num
        self.embedding_user = nn.Embedding(num_embeddings = self.num_users, embedding_dim = self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings = self.num_items, embedding_dim = self.factor_num)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user.weight, std = 0.01)
        nn.init.normal_(self.embedding_item.weight, std = 0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def get_embeddings(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        return user_embedding, item_embedding

class Generalized_Matrix_Factorization(Base_Module):
    def __init__(self, args, num_users, num_items):
        super(Generalized_Matrix_Factorization, self).__init__(args, num_users, num_items)
        self.affine_output = nn.Linear(in_features = self.factor_num, out_features = 1)

    def forward_vec(self, user_indices, item_indices):
        user_embedding, item_embedding = self.get_embeddings(user_indices, item_indices)
        vector = torch.mul(user_embedding, item_embedding)
        return vector
        
    def forward(self, user_indices, item_indices):
        element_product = self.forward_vec(user_indices, item_indices)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating.squeeze()

class Multi_Layer_Perceptron(Base_Module):
    def __init__(self, args, num_users, num_items, num_layers = 3):
        super(Multi_Layer_Perceptron, self).__init__(args, num_users, num_items)
        self.layers = args.layers
        self.affine_output = nn.Linear(in_features = self.layers[-1], out_features = 1)
        self.fc_layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.num_layers = num_layers
        
        for idx, (in_size, out_size) in enumerate(zip(
            self.layers[len(self.layers) - self.num_layers - 1 : len(self.layers) - 1], 
            self.layers[len(self.layers) - self.num_layers : len(self.layers)]
        )):
            self.fc_layers.append(nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())
            
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.affine_output.weight)
        
    def forward_vec(self, user_indices, item_indices):
        user_embedding, item_embedding = self.get_embeddings(user_indices, item_indices)
        
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            # vector = nn.BatchNorm1d()(vector)
            # vector = nn.Dropout(p=0.5)(vector)
        return vector
        
    def forward(self, user_indices, item_indices):
        vector = self.forward_vec(user_indices, item_indices)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()
        

class NeuMF(nn.Module):
    def __init__(self, args, num_users, num_items, num_layers = 3):
        super(NeuMF, self).__init__()
        self.dropout = args.dropout
        self.gmf_args = args
        self.gmf = Generalized_Matrix_Factorization(self.gmf_args, num_users, num_items)
        self.mlp_args = copy.deepcopy(args)
        self.mlp_args.factor_num = self.mlp_args.layers[len(self.mlp_args.layers) - num_layers - 1] // 2
        self.mlp = Multi_Layer_Perceptron(self.mlp_args, num_users, num_items, num_layers)
        self.affine_output = nn.Linear(in_features = self.mlp_args.layers[-1] + self.gmf_args.factor_num, out_features = 1)
        self.logistic = nn.Sigmoid()

    def init_weight(self):
        self.gmf.init_weight()
        self.mlp.init_weight()

    def forward(self, user_indices, item_indices):        
        mf_vector = self.gmf.forward_vec(user_indices, item_indices)
        mlp_vector = self.mlp.forward_vec(user_indices, item_indices)
        vector = torch.cat([0.5 * mlp_vector, 0.5 * mf_vector], dim = -1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()