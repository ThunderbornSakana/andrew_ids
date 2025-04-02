import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from models.mlp import MLP
from models.tabtransformer import TabTransformer

torch.manual_seed(88)
np.random.seed(88)

MODEL = "tabtransformer" # "mlp"
GPU_IDX = 0

nums = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

cats = [
    "protocol_type", "service", "flag", "land",
    "logged_in", "is_host_login", "is_guest_login"
]

cols = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

class KDDDataset(Dataset):
    def __init__(self, df, num_cols, cat_cols, label_col='target'):
        self.num_data = df[num_cols].values.astype(np.float32)
        self.cat_data = df[cat_cols].values.astype(np.int64)
        self.labels = df[label_col].values.astype(np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'numeric': torch.tensor(self.num_data[idx]),
            'categorical': torch.tensor(self.cat_data[idx]),
            'label': torch.tensor(self.labels[idx])
        }


def map_label(label):
    return 0 if label == "normal." else 1


def main():
    device = torch.device(f"cuda:{GPU_IDX}" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv("./datasets/kddcup.data.txt", header=None, names=cols)
    test_df = pd.read_csv("./datasets/corrected.txt", header=None, names=cols)

    train_df['target'] = train_df['label'].apply(map_label)
    test_df['target'] = test_df['label'].apply(map_label)

    cat_maps = {}
    for col in cats:
        unique_vals = train_df[col].unique().tolist()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        cat_maps[col] = mapping
        train_df[col] = train_df[col].map(mapping)
        test_df[col] = test_df[col].map(mapping)

    # Standardize numeric features.
    if MODEL == "mlp":
        scaler = StandardScaler()
        train_df[nums] = scaler.fit_transform(train_df[nums])
        test_df[nums] = scaler.transform(test_df[nums])


    if MODEL == "mlp":
        batch_size = 2560
        num_epochs = 30
        learning_rate = 0.003
        cat_embed_dim = 16
        hidden_dim = 128
        cat_dims = {col: train_df[col].nunique() for col in cats}
        model = MLP(num_numeric_features=len(nums),
                    cat_dims=cat_dims,
                    cat_embed_dim=cat_embed_dim,
                    hidden_dim=hidden_dim).to(device)
    elif MODEL == "tabtransformer":
        embed_dim = 16
        num_heads = 4
        num_layers = 2
        ffn_dim = 128
        mlp_hidden = 128
        num_epochs = 30
        batch_size = 2560
        learning_rate = 0.003
        cat_cardinalities = [int(train_df[col].nunique()) for col in cats]
        model = TabTransformer(cat_cardinalities=cat_cardinalities, 
                               num_numeric=len(nums), 
                               embed_dim=embed_dim, 
                               n_heads=num_heads, 
                               n_layers=num_layers, 
                               ffn_dim=ffn_dim, 
                               mlp_hidden=mlp_hidden).to(device)

    train_dataset = KDDDataset(train_df, nums, cats, label_col='target')
    test_dataset = KDDDataset(test_df, nums, cats, label_col='target')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):
            x_numeric = batch['numeric'].to(device)
            x_categorical = batch['categorical'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(x_numeric, x_categorical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_numeric.size(0)
        
        train_loss /= len(train_dataset)

        model.eval()
        all_labels, all_outputs, all_preds = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                x_numeric = batch['numeric'].to(device)
                x_categorical = batch['categorical'].to(device)
                labels = batch['label'].unsqueeze(1).to(device)
                outputs = model(x_numeric, x_categorical)
                predicted = (outputs > 0.5).float()
                all_labels.append(labels.cpu())
                all_outputs.append(outputs.cpu())
                all_preds.append(predicted.cpu())
        all_labels = torch.cat(all_labels).numpy().flatten()
        all_outputs = torch.cat(all_outputs).numpy().flatten()
        all_preds = torch.cat(all_preds).numpy().flatten()

        accuracy = (all_preds == all_labels).mean()
        roc_auc = roc_auc_score(all_labels, all_outputs)
        f1 = f1_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
