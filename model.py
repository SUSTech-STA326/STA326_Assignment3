import torch
import torch.nn as nn

class RecommenderModel(nn.Module):
    def __init__(self, config):
        super(RecommenderModel, self).__init__()
        self.config = config
        self.X = config["layers_num(X)"]
        self.model_type = config['model_type']

        self.embedding_user = torch.nn.Embedding(num_embeddings=config['num_users'], embedding_dim=config['latent_dim'])
        self.embedding_item = torch.nn.Embedding(num_embeddings=config['num_items'], embedding_dim=config['latent_dim'])

        if self.model_type == "GMF":
            self.linear_gmf = torch.nn.Linear(in_features=config['latent_dim'], out_features=1, bias = False)
        else:
            ### the mlp part is necessary for `NeuMF` and `MLP` model
            if self.config["layers_num(X)"] != 0:
                self.fc_layers = torch.nn.ModuleList()
                for idx, (in_size, out_size) in enumerate(zip(self.config['layers'][:-1], self.config['layers'][1:])):
                    self.fc_layers.append(torch.nn.Linear(in_size, out_size,bias = True))

            if self.model_type == "MLP":
                self.linear_mlp = torch.nn.Linear(in_features=self.config['layers'][-1], out_features=1, bias = False)
                self.linear_mlp0 = torch.nn.Linear(in_features=self.config['latent_dim']*2, out_features=1, bias = False)
            elif self.model_type == "NeuMF":
                self.linear_neumf = torch.nn.Linear(in_features=self.config['latent_dim']*2+self.X, out_features=1,bias = False)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, user_indices, item_indices):
        if self.model_type == 'GMF':
            user_embedding_gmf = self.embedding_user(user_indices)
            item_embedding_gmf = self.embedding_item(item_indices)
            vector = torch.mul(user_embedding_gmf, item_embedding_gmf)
            vector = self.linear_gmf(vector)
        else:
            #* initiate MLP part
            user_embedding_mlp = self.embedding_user(user_indices)
            item_embedding_mlp = self.embedding_item(item_indices)

            #* `MLP` or `NeuMF`
            if self.model_type == 'MLP':    ### only in `MLP` model, `X` can be 0
                vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
                if self.X != 0:
                    for fc_layer in self.fc_layers:
                        vector = fc_layer(vector)
                        vector = self.relu(vector)
                    vector = self.linear_mlp(vector)
                else:
                    vector = self.linear_mlp0(vector)
            elif self.model_type == 'NeuMF':
                ################# for gmf
                user_embedding_gmf = self.embedding_user(user_indices.long())
                item_embedding_gmf = self.embedding_item(item_indices.long())
                gmf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf)

                #################  for mlp
                mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
                for fc_layer in self.fc_layers:
                    mlp_vector = fc_layer(mlp_vector)
                    mlp_vector = self.relu(mlp_vector)
                mlp_vector = self.linear_mlp(mlp_vector)

                #################  concat mf and mlp vector
                vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
                vector = self.linear_neumf(vector)

        output = self.sigmoid(vector)
        return output.squeeze(1)
