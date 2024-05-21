import torch
import torch.nn as nn
from Models import MLP
from Models import GMF

def create_mlp_model(num_users, num_items, num_factors, num_hidden_layers):
    if num_hidden_layers == 0:
        layers = []  # No hidden layers
    else:
        layers = [num_factors] * num_hidden_layers  # Same size as latent factors for simplicity
    return MLP(num_users, num_items, layers)

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors, mlp_layers):
        super(NeuMF, self).__init__()
        self.gmf = GMF(num_users, num_items, num_factors)
        self.mlp = create_mlp_model(num_users, num_items, num_factors, 4)
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