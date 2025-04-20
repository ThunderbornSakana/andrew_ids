import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_numeric_features, cat_dims, cat_embed_dim, hidden_dim):
        super(CNNModel, self).__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_embeddings=cat_dims[col], embedding_dim=cat_embed_dim)
            for col in cat_dims.keys()
        })
        
        # CNN for numeric input (reshape to [B, C=1, L])
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(4)  # reduce to fixed size

        # Final MLP
        cat_embed_total = len(cat_dims) * cat_embed_dim
        conv_out_dim = 16 * 4  # 16 channels × 4 pooled length
        final_input_dim = cat_embed_total + conv_out_dim
        
        self.fc1 = nn.Linear(final_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_numeric, x_categorical):
        # CNN over numeric input
        x_numeric = x_numeric.unsqueeze(1)  # (B, 1, 34)
        x = self.relu(self.conv1(x_numeric))
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # (B, 16, 4)
        x = x.view(x.size(0), -1)  # flatten to (B, 64)

        # Embedding for categorical features
        embeds = [self.embeddings[col](x_categorical[:, i]) for i, col in enumerate(self.embeddings.keys())]
        x_cat = torch.cat(embeds, dim=1)

        # Combine and pass through MLP
        x = torch.cat([x, x_cat], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)
