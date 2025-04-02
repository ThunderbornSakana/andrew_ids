import torch
import torch.nn as nn


class TabTransformer(nn.Module):
    def __init__(self, cat_cardinalities, num_numeric, embed_dim, 
                 n_heads, n_layers, ffn_dim, mlp_hidden):
        super(TabTransformer, self).__init__()
        self.embed_layers = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities
        ])
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, 
                                                  dim_feedforward=ffn_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.num_norm = nn.LayerNorm(num_numeric)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * len(cat_cardinalities) + num_numeric, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_nums, x_cats):
        # 1. Embed each categorical feature.
        embedded_feats = []
        for i, embed in enumerate(self.embed_layers):
            embedded_feats.append(embed(x_cats[:, i]))

        x_cat = torch.stack(embedded_feats, dim=1)
        # 2. Transformer encoder on categorical embeddings.
        x_cat_transformed = self.transformer(x_cat)
        # 3. Flatten all transformed categorical tokens.
        batch_size = x_cat_transformed.size(0)
        x_cat_flat = x_cat_transformed.reshape(batch_size, -1)
        # 4. Layer-normalize numeric features.
        x_num_norm = self.num_norm(x_nums)
        # 5. Concatenate transformed categorical and numeric features.
        combined = torch.cat([x_cat_flat, x_num_norm], dim=1)
        # 6. MLP for final prediction.
        logits = self.mlp(combined)
        prob = self.sigmoid(logits)
        return prob
