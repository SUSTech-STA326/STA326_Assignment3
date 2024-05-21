import torch
import torch.nn as nn

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