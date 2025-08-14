import networkx as nx
import pandas as pd
from collections import defaultdict
import numpy as np
from typing import Dict, Any, NamedTuple, Tuple
from torch_geometric.data import Data
import torch
import os

class ProcessedData(NamedTuple):
    node_dict: Dict[int, Dict[int, Dict[str, Any]]]
    graph_dict: Dict[int, Dict[int, nx.Graph]]
    pyg_graphs: list[Data]
    pyg_node_map: Dict[Tuple[int, int], Dict[str, int]]

class GraphProcessor:
    def __init__(self, graph_configs, processed_path: str = "data/processed"):
        self.processed_path = processed_path
        self.graph_configs = graph_configs
        self.graphs = {}
    
    def build_node_features(self, embeddings_source: pd.DataFrame, pairs: pd.DataFrame, configs: Dict[str, Any]) -> Tuple[Dict[int, Dict[int, Dict[str, np.ndarray]]], int]:
        """Pool embeddings per (subreddit_id, timestep, author) from [comments] data"""
        def _pool(arrs):
            arrs = np.stack(arrs, axis=0)
            if pooling == 'mean':
                return arrs.mean(axis=0)
            elif pooling == 'sum':
                return arrs.sum(axis=0)
            elif pooling == 'max':
                return arrs.max(axis=0)
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")
        
        pooling = configs.get('pooling', 'mean')
        print(f"Building node features with pooling: {pooling}")
        
        # 1. Collect all authors from user pairs
        all_users = pd.concat([
            pairs[['subreddit_id','timestep','src_author']].rename(columns={'src_author': 'author'}),
            pairs[['subreddit_id','timestep','dst_author']].rename(columns={'dst_author': 'author'})
        ], ignore_index=True).drop_duplicates()
        print(f"    + Total unique authors in pairs: {len(all_users)}")

        # 2. Filter embeddings to only those authors & snapshots
        embs_df = embeddings_source[['subreddit_id','timestep','author','embeddings']]
        embs_df = embs_df.merge(all_users, on=['subreddit_id','timestep','author'], how='inner')
        embs_df['embeddings'] = embs_df['embeddings'].map(
            lambda x: x if isinstance(x, np.ndarray) else np.asarray(x, dtype=float)
        )

        # 3. Pool within each (subreddit, timestep, author)
        pooled = embs_df.groupby(['subreddit_id','timestep','author'], sort=False)['embeddings'].agg(_pool)
        emb_dim = pooled.iloc[0].shape[0] if not pooled.empty else 0
        print(f"    + Total pooled vectors: {len(pooled)}")
        print(f"    + Pooled vector dimension: {emb_dim}")

        # 4. Convert to nested dict
        node_dict: Dict[int, Dict[int, Dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
        for index, vec in pooled.items():
            sub, ts, author = index     # type: ignore
            node_dict[int(sub)][int(ts)][str(author)] = vec
        
        return {sub: dict(ts_dict) for sub, ts_dict in node_dict.items()}, emb_dim
    
    def build_graph_snapshots(self, pairs: pd.DataFrame, node_dict: Dict[int, Dict[int, Dict[str, np.ndarray]]], configs: Dict[str, Any]) -> Dict[int, Dict[int, nx.Graph]]:
        # {subreddit_id: {timestep: graph}}
        directed = configs.get('directed', True)
        use_wcc = configs.get('use_wcc', False)
        edge_attrs = configs.get('edge_attrs', ['mean_confidence', 'net_vector'])
        print(f"Building graph snapshots: directed={directed}, use_wcc={use_wcc}, edge_attrs={edge_attrs}")
        graph_dict = {}
        
        # Build from pairs
        for (subreddit_id, timestep), group in pairs.groupby(['subreddit_id', 'timestep']):
            subreddit_id = int(subreddit_id)
            timestep = int(timestep)

            G = nx.DiGraph() if directed else nx.Graph()

            # Add nodes with features
            ts_nodes = node_dict.get(subreddit_id, {}).get(timestep, {})
            for author, embedding in ts_nodes.items():
                G.add_node(author, embedding=embedding)

            # Add edges
            for _, row in group.iterrows():
                src = row['src_author']
                dst = row['dst_author']
                if src in G and dst in G:
                    edge_data = {attr: row[attr] for attr in edge_attrs if attr in row}
                    G.add_edge(src, dst, **edge_data)
            
            # Apply weakly connected component filtering if requested
            if use_wcc and len(G.nodes()) > 0:
                edges_before_wcc = len(G.edges())
                if directed:
                    # For directed graphs, get largest weakly connected component
                    wcc_components = list(nx.weakly_connected_components(G))
                else:
                    # For undirected graphs, get largest connected component
                    wcc_components = list(nx.connected_components(G))
                
                if wcc_components:
                    # Get the largest component
                    largest_component = max(wcc_components, key=len)
                    G = G.subgraph(largest_component).copy()
                    
                    # Log filtered
                    edges_after_wcc = len(G.edges())
                    edges_filtered = edges_before_wcc - edges_after_wcc
                    if edges_filtered > 0:
                        print(f"    + [Subreddit {subreddit_id}, T{timestep}] {edges_filtered} edges filtered by WCC ({edges_before_wcc} -> {edges_after_wcc})")
            
            graph_dict.setdefault(subreddit_id, {})[timestep] = G

        return graph_dict

    def build_pyg_graphs(self, graph_dict) -> tuple[list, dict]:
        """Convert NetworkX graphs to PyG Data objects"""
        print("Converting NetworkX graphs to PyG data objects")
        pyg_graphs = []
        master_node_map = {}  # {(subreddit_id, timestep): {node_name: idx}}
        
        # Sort by subreddit_id and timestep for consistent ordering
        for sub in sorted(graph_dict.keys()):
            ts_dict = graph_dict[sub]
            for ts in sorted(ts_dict.keys()):
                G = ts_dict[ts]
                if len(G.nodes()) == 0:
                    continue    # Skip empty graphs
                
                node_list = sorted(G.nodes())
                node_map = {node: idx for idx, node in enumerate(node_list)}
                master_node_map[(sub, ts)] = node_map

                # Extract node features
                node_features = []
                for node in node_list:
                    embedding = G.nodes[node].get('embedding')
                    if embedding is not None:
                        node_features.append(embedding)
                    else:
                        node_features.append(np.zeros(384))  # Default to zero vector if no embedding
                x = torch.tensor(node_features, dtype=torch.float)

                # Extract edges
                edge_list, edge_attrs = [], []
                for src, dst, data in G.edges(data=True):
                    src_idx = node_map[src]
                    dst_idx = node_map[dst]
                    edge_list.append([src_idx, dst_idx])

                    # Extract edge attributes
                    edge_attr = []
                    for attr in ['mean_confidence', 'net_vector']:
                        if attr in data:
                            if attr == 'net_vector' and isinstance(data[attr], (list, np.ndarray)):
                                edge_attr.extend(data[attr])  # Flatten vector attributes
                            else:
                                edge_attr.append(data[attr])
                    edge_attrs.append(edge_attr)
            
                # Convert to tensors (moved inside the timestep loop)
                if edge_list:
                    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0, len(['mean_confidence', 'net_vector'])), dtype=torch.float)
                
                # Create PyG data object (moved inside the timestep loop)
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(node_list),
                    num_edges=len(edge_list),
                    node_map=node_map,
                    subreddit_id=sub,
                    local_timestep=ts
                )
                pyg_graphs.append(data)
        
        print(f"    + Created {len(pyg_graphs)} PyG graphs")
        return pyg_graphs, master_node_map

    def run(self, pairs: pd.DataFrame, embeddings_source: pd.DataFrame) -> ProcessedData:
        """Process data to create node features and graph snapshots"""
        # 1. Build node features
        node_cfg = self.graph_configs.get('node_features', {})
        node_dict, vec_dim = self.build_node_features(embeddings_source, pairs, node_cfg)
        
        # 2. Build graph snapshots
        construction_cfg = self.graph_configs.get('construction', {})
        graph_dict = self.build_graph_snapshots(pairs, node_dict, construction_cfg)

        # 3. Build PyG graphs
        pyg_graphs, pyg_node_map = self.build_pyg_graphs(graph_dict)
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

        # 4. Save PyG graphs to file
        torch.save(pyg_graphs, os.path.join(self.processed_path, f'pyg_graphs_{vec_dim}D.pt'))
        print(f"Saved PyG graphs to {self.processed_path}/pyg_graphs_{vec_dim}D.pt")
        
        return ProcessedData(
            node_dict=node_dict, 
            graph_dict=graph_dict, 
            pyg_graphs=pyg_graphs,
            pyg_node_map=pyg_node_map
        )
