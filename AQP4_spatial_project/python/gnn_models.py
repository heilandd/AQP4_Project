import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class CustomGATConv(GATConv):
    \"\"\"GAT layer that gracefully handles scalar edge_attr after batching.\"\"\"
    def __init__(self, in_channels, out_channels, heads=1, concat=True, edge_dim=1, **kwargs):
        super().__init__(in_channels, out_channels, heads=heads, concat=concat,
                         edge_dim=edge_dim, **kwargs)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        if edge_attr is not None and edge_attr.dim() == 2 and edge_attr.shape[1] == 1:
            edge_attr = edge_attr.squeeze(1)

        if return_attention_weights:
            out, attn_weights = super().forward(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            return out, attn_weights
        else:
            return super().forward(x, edge_index, edge_attr=edge_attr)


class GraphEncoder(nn.Module):
    \"\"\"Two-layer GAT encoder that returns node- and graph-level embeddings.\"\"\"

    def __init__(self, num_features_exp, hidden_channels, edge_dim=1, num_heads=5):
        super().__init__()
        per_head_hidden = hidden_channels // num_heads
        self.conv1 = CustomGATConv(num_features_exp, per_head_hidden,
                                   heads=num_heads, edge_dim=edge_dim)
        self.conv2 = CustomGATConv(per_head_hidden * num_heads, per_head_hidden,
                                   heads=num_heads, edge_dim=edge_dim)
        self.bn1 = nn.LayerNorm(hidden_channels)
        self.bn2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(0.5)
        self.merge = nn.Linear(hidden_channels, hidden_channels)
        nn.init.xavier_uniform_(self.merge.weight.data)

    def forward(self, x, edge_index, edge_attr, batch):
        x, att1 = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        x = F.leaky_relu(x)
        x = self.dropout(self.bn1(x))
        x, att2 = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        x = F.leaky_relu(x)
        x = self.dropout(self.bn2(x))
        x = self.merge(x)
        x = F.leaky_relu(x)

        x_pooled = global_mean_pool(x, batch)

        return x, x_pooled, att1, att2


class MLP(nn.Module):
    \"\"\"Simple MLP classification head.\"\"\"

    def __init__(self, hidden_channels, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.model(x)


class RegressionHead(nn.Module):
    \"\"\"Regression head that maps graph embeddings to a scalar.\"\"\"

    def __init__(self, hidden, dropout=0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )
        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.mlp(x).squeeze(1)


class GraphMERFISHRegress(nn.Module):
    \"\"\"Multi-task GNN: regression + classification.\"\"\"

    def __init__(self, in_feat, hidden, num_classes_task1, edge_dim=1):
        super().__init__()
        self.encoder = GraphEncoder(in_feat, hidden, edge_dim=edge_dim)
        self.head = RegressionHead(hidden)
        self.task1_head = MLP(hidden, num_classes_task1)

    def forward(self, data):
        _, graph_latent, att1, att2 = self.encoder(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        pred_reg = self.head(graph_latent)
        pred_cls = self.task1_head(graph_latent)
        return graph_latent, pred_cls, pred_reg, att1, att2
