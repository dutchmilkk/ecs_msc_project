"""
This implementation is based on:
Alatawi, F., Sheth, P., & Liu, H. "Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach."
In Proceedings of ASONAM '23: International Conference on Advances in Social Networks Analysis and Mining, 2023.

DOI: 10.1145/3625007.3627731
GitHub: https://github.com/faalatawi/echo-chamber-score/blob/main/src/EchoGAE.py
"""


import numpy as np
import networkx as nx
import pandas as pd

import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges

from typing import Tuple
from pandas import DataFrame
from networkx import Graph

from .GAE import run

# import logging

def EchoGAE_algorithm(
    G,
    user_embeddings=None,
    show_progress=True,
    epochs=300,
    hidden_channels=100,
    out_channels=50,
    seed=42
) -> Tuple[np.ndarray, torch.nn.Module, list, list, list]:
    
    # logger = logging.getLogger(__name__)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Create node features
    if user_embeddings is None:
        X = torch.eye(len(G.nodes), dtype=torch.float32, device=DEVICE)
    else:
        X = []
        for node in G.nodes:
            X.append(user_embeddings[node])
        X = np.array(X)
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    # Create edge list
    edge_list = np.array(G.edges).T
    edge_list = torch.tensor(edge_list, dtype=torch.int64).to(DEVICE)

    # Data object
    data = Data(x=X, edge_index=edge_list)
    data = train_test_split_edges(data)

    # Run the model
    model, x, train_pos_edge_index, losses, aucs, aps = run(
        data,
        show_progress=show_progress,
        epochs=epochs,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        seed=seed
    )

    # Embedding
    GAE_embedding = model.encode(x, train_pos_edge_index).detach().cpu().numpy()

    # [2025.07.26 - VD] Added return values for debug
    return GAE_embedding, model, losses, aucs, aps
