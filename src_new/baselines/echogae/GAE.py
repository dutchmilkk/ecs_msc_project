"""
This implementation is based on:
Alatawi, F., Sheth, P., & Liu, H. "Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach."
In Proceedings of ASONAM '23: International Conference on Advances in Social Networks Analysis and Mining, 2023.

DOI: 10.1145/3625007.3627731
GitHub: https://github.com/faalatawi/echo-chamber-score/blob/main/src/GAE.py
"""

import torch
from torch_geometric.nn import GCNConv, GAE


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()

        # Layer 1:
        # cached only for transductive learning
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)

        # Layer 2:
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def __train(model, optimizer, x, train_pos_edge_index):
    model.train()
    optimizer.zero_grad()

    z = model.encode(x, train_pos_edge_index)

    # Compute loss
    loss = model.recon_loss(z, train_pos_edge_index)

    loss.backward()

    # Update parameters
    optimizer.step()

    return float(loss)


def __test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def run(data, hidden_channels, out_channels=10, epochs=300, show_progress=True, seed=42):
    # set the seed
    torch.manual_seed(seed)

    num_features = data.num_features

    # model
    model = GAE(GCNEncoder(num_features, hidden_channels, out_channels))

    # move to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)

    # inizialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # [2025.07.26 - VD] Add tracker for losses, auc, ap
    losses, aucs, aps = [], [], []

    for epoch in range(1, epochs + 1):
        loss = __train(model, optimizer, x, train_pos_edge_index)

        auc, ap = __test(
            model,
            x,
            train_pos_edge_index,
            data.test_pos_edge_index,
            data.test_neg_edge_index,
        )
        if show_progress:
            print("Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}".format(epoch, auc, ap))

        # [2025.07.26 - VD] Track losses, auc, ap
        losses.append(loss)
        aucs.append(auc)
        aps.append(ap)

    # [2025.07.26 - VD] Add additional return values
    return model, x, train_pos_edge_index, losses, aucs, aps
