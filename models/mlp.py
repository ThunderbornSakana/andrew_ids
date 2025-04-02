import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_numeric_features, cat_dims, cat_embed_dim, hidden_dim):
        super(MLP, self).__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_embeddings=cat_dims[col], embedding_dim=cat_embed_dim)
            for col in cat_dims.keys()
        })
        input_dim = num_numeric_features + len(cat_dims) * cat_embed_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_numeric, x_categorical):
        embeds = []
        for i, col in enumerate(self.embeddings.keys()):
            embeds.append(self.embeddings[col](x_categorical[:, i]))
        x_cat = torch.cat(embeds, dim=1)
        
        x = torch.cat([x_numeric, x_cat], dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)
