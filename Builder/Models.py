import torch
import torch.nn as nn
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


def MLP_with_hidden_layers(num_users, num_items, num_factors, num_hidden_layers):
    if num_hidden_layers == 0:
        layers = []  # No hidden layers
    else:
        layers = [num_factors] * num_hidden_layers  # Same size as latent factors for simplicity
    return MLP(num_users, num_items, layers)


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


class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers):
        super(MLP, self).__init__()
        self.user_emb = nn.Embedding(num_users, layers[0] // 2 if layers else 8)  # default to 8 if no layers
        self.item_emb = nn.Embedding(num_items, layers[0] // 2 if layers else 8)  # default to 8 if no layers
        self.fc_layers = nn.ModuleList()
        for idx in range(1, len(layers)):
            self.fc_layers.append(nn.Linear(layers[idx - 1], layers[idx]))
        self.output_layer = nn.Linear(layers[-1] if layers else 16, 1)  # default to 16 if no layers

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_emb(user_ids)
        item_embedding = self.item_emb(item_ids)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for layer in self.fc_layers:
            vector = torch.relu(layer(vector))
        output = self.output_layer(vector)
        return output.sigmoid()


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors, mlp_layers):
        super(NeuMF, self).__init__()
        self.gmf = GMF(num_users, num_items, num_factors)
        self.mlp = MLP_with_hidden_layers(num_users, num_items, num_factors, 4)
        #self.output_layer = nn.Linear(num_factors + mlp_layers[-1], 1)
        self.output_layer = nn.Linear(2, 1)  # Changed from num_factors + mlp_layers[-1] to 2

    def forward(self, user_ids, item_ids):
        gmf_output = self.gmf(user_ids, item_ids)
        mlp_output = self.mlp(user_ids, item_ids)

        # Ensure both outputs have the same dimensions
        gmf_output = gmf_output.unsqueeze(1)  # Change shape from [batch_size] to [batch_size, 1]
        concatenated_output = torch.cat((gmf_output, mlp_output), dim=-1)  # Concatenate along the last dimension

        output = self.output_layer(concatenated_output)
        return output.sigmoid()
