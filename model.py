import torch
import torch.nn as nn

class RecommenderModel(nn.Module):
    def __init__(self, config):
        super(RecommenderModel, self).__init__()
        self.model_type = config['model_type']

        if self.model_type == "GMF":
            self.embedding_user_gmf = torch.nn.Embedding(num_embeddings=config['num_users'], embedding_dim=config['embedding_dim_mf'])
            self.embedding_item_gmf = torch.nn.Embedding(num_embeddings=config['num_items'], embedding_dim=config['embedding_dim_mf'])
            self.linear_gmf = torch.nn.Linear(in_features=config['embedding_dim_mf'], out_features=1, bias = False)
        else:
            ### the mlp part is necessary for both `NeuMF` and `MLP` model
            self.X = config['mlp_layers(X)']
            self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=config['num_users'], embedding_dim=int((config['mlp_layers'][0])/2))
            self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=config['num_items'], embedding_dim=int((config['mlp_layers'][0])/2))

            if config["mlp_layers(X)"] != 0:
                self.fc_layers = torch.nn.ModuleList()
                for idx, (in_size, out_size) in enumerate(zip(config['mlp_layers'][:-1], config['mlp_layers'][1:])):
                    self.fc_layers.append(torch.nn.Linear(in_size, out_size, bias = True))
            else:
                self.linear_mlp0 = torch.nn.Linear(in_features=(config['mlp_layers'][0]), out_features=1, bias = False)

            if self.model_type == "MLP" and config["mlp_layers(X)"] != 0:
                self.linear_mlp = torch.nn.Linear(in_features=config['mlp_layers'][-1], out_features=1, bias = False)

            if self.model_type == "NeuMF":
                self.embedding_user_gmf = torch.nn.Embedding(num_embeddings=config['num_users'], embedding_dim=config['embedding_dim_mf'])
                self.embedding_item_gmf = torch.nn.Embedding(num_embeddings=config['num_items'], embedding_dim=config['embedding_dim_mf'])
                self.linear_neumf = torch.nn.Linear(in_features=config['embedding_dim_mf']+config['mlp_layers'][-1], out_features=1,bias = False)

        # self.dropout_mf = torch.nn.Dropout(p=config['dropout'])
        # self.dropout_mlp = torch.nn.Dropout(p=config['dropout'])
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, user_indices, item_indices):
        if self.model_type == 'GMF':
            user_embedding_gmf = self.embedding_user_gmf(user_indices)
            item_embedding_gmf = self.embedding_item_gmf(item_indices)
            vector = torch.mul(user_embedding_gmf, item_embedding_gmf)
            vector = self.linear_gmf(vector)
        else:
            #* initiate MLP part
            user_embedding_mlp = self.embedding_user_mlp(user_indices)
            item_embedding_mlp = self.embedding_item_mlp(item_indices)

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
                user_embedding_gmf = self.embedding_user_gmf(user_indices)
                item_embedding_gmf = self.embedding_item_gmf(item_indices)
                gmf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf)

                #################  for mlp
                mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
                for fc_layer in self.fc_layers:
                    mlp_vector = fc_layer(mlp_vector)
                    mlp_vector = self.relu(mlp_vector)

                #################  concat mf and mlp vector
                # print(gmf_vector.size(), mlp_vector.size())
                vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
                vector = self.linear_neumf(vector)

        output = self.sigmoid(vector)
        return output.squeeze(1)
        # return output
