import torch
import numpy as np
import torch.nn.functional as F


class NeuralCollaborativeFiltering(torch.nn.Module):
    def __init__(self, field_dims, user_field_idx, item_field_idx, embed_dim, mlp_dims, dropout, mode = "NCF"):
        super().__init__()
        #  mlp_dims: 中间层的维数，比如(16, 16)表示有两个中间层，每个中间层的维数是16
        self.mode = mode
        self.user_field_idx = user_field_idx # 0
        self.item_field_idx = item_field_idx # 1
        self.embedding = FeaturesEmbedding(field_dims, embed_dim) # 索引到embedding
        self.embed_output_dim = len(field_dims) * embed_dim # 对于mlp是先做contatenate，所以是2*16
        if self.mode == "NCF":
            self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
            self.fc = torch.nn.Linear(mlp_dims[-1] + embed_dim, 1)
        elif self.mode == "MLP":
            self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        elif self.mode == "GMF":
            self.fc = torch.nn.Linear(embed_dim, 1)
            
    def forward(self, x):
        index = x
        x = self.embedding(x) # 获得embedding，维度为embed_output_dim
        user_x = x[:, self.user_field_idx].squeeze(1) # 按照0取出user的embedding
        item_x = x[:, self.item_field_idx].squeeze(1) # 按照1取出item的embedding
        gmf = user_x * item_x # generalized matrix factorization
        if self.mode == "NCF":
            x = self.mlp(x.view(-1, self.embed_output_dim)) # 将embedding展平，送入mlp
            x = torch.cat([gmf, x], dim=1) # 将gmf和mlp的结果拼接
            x = self.fc(x).squeeze(1) 
        elif self.mode == "MLP":
            x = self.mlp(x.view(-1, self.embed_output_dim))
            x = x.squeeze(1) # 最后一层直接输出
        elif self.mode == "GMF":
            x = self.fc(gmf).squeeze(1)
        return torch.sigmoid(x), index
    
class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim) #6040+3706 为每个user和item生成一个embed_dim维(默认为16)的向量
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64) 
        # [0, 6040] 划分开始索引，在用数据的时候加上这个偏移量，就可以找到对应的embedding
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)# 用xavier_uniform_初始化embedding的权重

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0) # 加上偏移量，方便索引
        return self.embedding(x)
    
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            if embed_dim != 0:
                layers.append(torch.nn.Linear(input_dim, embed_dim))
                layers.append(torch.nn.BatchNorm1d(embed_dim))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(p=dropout))
                input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)