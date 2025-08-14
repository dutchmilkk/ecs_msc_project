import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, NamedTuple
from collections import defaultdict

from src.utils.edge_weights import compute_edge_weight

class CommunityDetectionResult(NamedTuple):
    meta: Dict[str, Any]
    partitions: pd.DataFrame
    labels_array: Dict[int, Dict[int, np.ndarray]]
    labels_index_dict: Dict[int, Dict[int, Dict[int, int]]]
    labels_name_dict: Dict[int, Dict[int, Dict[str, int]]]
    optimization_scans: pd.DataFrame | None
    optimization_best: pd.DataFrame | None
    optimization_summary: pd.DataFrame | None

class LeidenCommunityProcessor:
    def __init__(self, community_configs: Dict):
        self.community_configs = community_configs
        self.seed = community_configs.get('seed', 42)
        self.optimization_cfg = community_configs.get('optimization', {})
        self.algorithm = community_configs.get('algorithm', 'leiden')
        if self.algorithm != 'leiden':
            raise NotImplementedError("LeidenCommunityProcessor only supports 'leiden' algorithm.")
    
    def _convert_nx_to_igraph(
        self, G: nx.Graph, 
        weight_strategy: str | None = None
    ) -> Tuple[ig.Graph, Dict[str, int], Dict[int, str]]:
        # 1. Get edges (with data) and nodes
        edges_with_data = list(G.edges(data=True))
        nodes = sorted(G.nodes())

        # 2. Create igraph graph
        ig_graph = ig.Graph()
        ig_graph.add_vertices(len(nodes))

        # 3. Map node names to indices
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}

        # 4. Add edges (strip data part)
        edge_indices = [(node_to_idx[src], node_to_idx[dst]) for src, dst, _ in edges_with_data]
        ig_graph.add_edges(edge_indices)
        ig_graph.vs['name'] = nodes

        # 5. Add weights if specified
        if weight_strategy:
            edge_weights = []
            for src, dst, data in edges_with_data:
                weight = compute_edge_weight(data, weight_strategy)
                edge_weights.append(weight)
            min_weight = min(edge_weights)
            if min_weight <= 0:
                offset = abs(min_weight) + 1.0
                edge_weights = [w + offset for w in edge_weights]
            ig_graph.es['weight'] = edge_weights

        return ig_graph, node_to_idx, idx_to_node

    def _get_community_labels(
        self, partition, 
        idx_to_node: Dict[int, str] | None= None, 
        return_by_idx: bool=True
    ) -> Tuple[Dict, np.ndarray]:
        """
        Return community labels for the graph.
        Args:
            partition: The community partition object.
            idx_to_node: Mapping from node indices to node names.
            return_by_idx: Whether to return labels by node index or name.
        Returns:
            Tuple[Dict, np.ndarray]: A tuple containing the labels dictionary and the labels array.
        """
        # 1. Create numpy array from partition membership (always indexed by node position)
        labels_arr = np.array(partition.membership)
        
        # 2. Return labels by selected method
        if return_by_idx:
            # Return labels by node index
            labels_dict = {}
            for node_idx, comm_id in enumerate(partition.membership):
                labels_dict[node_idx] = comm_id
            return labels_dict, labels_arr
        else:
            # Return labels by node name
            if idx_to_node is None:
                raise ValueError("`idx_to_node` mapping required when return_by_idx=False")
            labels_dict = {idx_to_node[node_idx]: comm_id for node_idx, comm_id in enumerate(partition.membership)}
            return labels_dict, labels_arr

    def analyze_optimal_community_parameters(
        self, 
        graph_dict: Dict[int, Dict[int, nx.Graph]]
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """
        Scan resolution grid over each graph:
          - Converts each NetworkX graph once to igraph (with optional weights)
          - Runs Leiden for each resolution (if partition type supports it)
          - Applies constraints (min communities, min community size, fragmentation guard)
          - Optional early stopping if metric stabilizes
          - Selects best resolution per graph or a uniform resolution across graphs
          - Builds detailed scan, best-selection, and summary DataFrames

        Returns:
          meta (dict), scans_df (pd.DataFrame), best_df (pd.DataFrame), summary_df (pd.DataFrame or None)
        """
        #=========================================
        # HELPER FUNCTIONS
        #=========================================
        # Helper to run partition (conditionally pass resolution parameter)
        def _run_partition(ig_graph: ig.Graph, resolution: float | None = None):
            kwargs = {
                'weights': 'weight' if 'weight' in ig_graph.es.attributes() else None,
                'seed': self.seed
            }
            if resolution is not None:
                try:
                    return la.find_partition(ig_graph, partition_cls, resolution_parameter=resolution, **kwargs)
                except TypeError:
                    # Partition type does not accept resolution_parameter
                    return la.find_partition(ig_graph, partition_cls, **kwargs)
            return la.find_partition(ig_graph, partition_cls, **kwargs)
        
        # Evaluate a partition -> dict row
        def _eval_partition(G: nx.Graph, partition, resolution: float | None) -> dict:
            sizes = np.bincount(partition.membership)
            row = {
                'resolution': resolution,
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'num_communities': int(len(sizes)),
                'min_size': int(sizes.min()) if len(sizes) else 0,
                'max_size': int(sizes.max()) if len(sizes) else 0,
                'mean_size': float(sizes.mean()) if len(sizes) else 0.0,
                'metric_value': partition.modularity if metric == 'modularity' else partition.modularity,
                'modularity': partition.modularity
            }
            return row
        
        #=========================================
        # END HELPER FUNCTIONS
        #=========================================
        import math

        # Optimization configuration
        cfg = self.optimization_cfg
        mode = cfg.get('mode', 'per_graph')              # 'per_graph' or 'uniform'
        metric = cfg.get('metric', 'modularity')         # currently only modularity implemented
        min_comms = cfg.get('min_communities', 2)
        min_comm_size = cfg.get('min_community_size', 3)
        max_comm_factor = cfg.get('max_communities_factor', 4)
        res_cfg = cfg.get('resolution', {})
        res_grid = res_cfg.get('grid', [1.0])
        early_stop = res_cfg.get('early_stop', {'window': 3, 'delta': 0.001})
        w_window = early_stop.get('window', 3)
        w_delta = early_stop.get('delta', 0.001)

        # Leiden-specific config
        weight_strategy = self.community_configs.get('weights', {}).get('strategy', 'agreement_diff')
        algo_cfg = self.community_configs.get('algorithms', {}).get(self.algorithm, {})
        partition_type_name = algo_cfg.get('partition_type', 'RBConfigurationVertexPartition')

        partition_cls_map = {
            'RBConfigurationVertexPartition': la.RBConfigurationVertexPartition,
            'ModularityVertexPartition': la.ModularityVertexPartition,
            'CPMVertexPartition': la.CPMVertexPartition
        }
        partition_cls = partition_cls_map.get(partition_type_name, la.RBConfigurationVertexPartition)

        # Cache igraph conversions (apply weights once per graph)
        ig_cache = {}
        for sub_id, ts_dict in graph_dict.items():
            for ts, G in ts_dict.items():
                if G.number_of_nodes() == 0:
                    continue
                ig_graph, _, _ = self._convert_nx_to_igraph(G, weight_strategy=weight_strategy)
                ig_cache[(sub_id, ts)] = ig_graph

        per_graph_results = {}          # (sub_id, ts) -> {'scans': [...], 'best': row or None}
        uniform_resolution_scores = []  # aggregated resolution performance (if mode == uniform)

        # Scan resolutions per graph
        for (sub_id, ts), ig_graph in ig_cache.items():
            G = graph_dict[sub_id][ts]
            scans = []
            recent = []
            best_row = None
            n = G.number_of_nodes()

            for r in res_grid:
                # Skip passing resolution if partition class ignores it (Modularity)
                resolution_arg = r if partition_cls is not la.ModularityVertexPartition else None
                partition = _run_partition(ig_graph, resolution=resolution_arg)
                row = _eval_partition(G, partition, r)
                scans.append(row)

                # Constraints check
                constraint_ok = (row['num_communities'] >= min_comms and row['min_size'] >= min_comm_size)
                if constraint_ok:
                    if best_row is None or row['metric_value'] > best_row['metric_value']:
                        best_row = row

                # Early stopping window
                recent.append(row['metric_value'])
                if len(recent) > w_window:
                    recent.pop(0)
                    if (max(recent) - min(recent)) < w_delta:
                        break

                # Fragmentation guard
                if row['num_communities'] > max_comm_factor * math.sqrt(n):
                    break

            per_graph_results[(sub_id, ts)] = {'scans': scans, 'best': best_row}

        # Uniform mode: choose single resolution by highest mean metric across graphs
        uniform_choice = None
        if mode == 'uniform':
            agg = {}
            for data in per_graph_results.values():
                for row in data['scans']:
                    agg.setdefault(row['resolution'], []).append(row['metric_value'])
            res_scores = [
                {'resolution': r, 'mean_metric': float(np.mean(vals)), 'std_metric': float(np.std(vals))}
                for r, vals in agg.items()
            ]
            if res_scores:
                res_scores.sort(key=lambda d: d['mean_metric'], reverse=True)
                uniform_choice = res_scores[0]['resolution']
                uniform_resolution_scores = res_scores

            # Override best rows (respect constraints)
            if uniform_choice is not None:
                for data in per_graph_results.values():
                    candidates = [
                        row for row in data['scans']
                        if row['resolution'] == uniform_choice
                        and row['num_communities'] >= min_comms
                        and row['min_size'] >= min_comm_size
                    ]
                    if candidates:
                        data['best'] = max(candidates, key=lambda r: r['metric_value'])

        # Flatten scan & best results
        scan_rows, best_rows = [], []
        for (sub_id, ts), data in per_graph_results.items():
            for row in data['scans']:
                scan_rows.append({'subreddit_id': sub_id, 'timestep': ts, **row})
            if data['best']:
                best_rows.append({'subreddit_id': sub_id, 'timestep': ts, **data['best']})

        scans_df = pd.DataFrame(scan_rows) if scan_rows else pd.DataFrame()
        best_df = pd.DataFrame(best_rows) if best_rows else pd.DataFrame()

        summary_df = None
        if not scans_df.empty:
            scans_df['constraints_ok'] = (
                (scans_df['num_communities'] >= min_comms) &
                (scans_df['min_size'] >= min_comm_size)
            )
            constrained = scans_df[scans_df['constraints_ok']]
            base_df = constrained if not constrained.empty else scans_df
            summary_df = (
                base_df.groupby('resolution')
                .agg(
                    metric_mean=('metric_value', 'mean'),
                    metric_std=('metric_value', 'std'),
                    modularity_mean=('modularity', 'mean'),
                    num_communities_mean=('num_communities', 'mean'),
                    num_communities_min=('num_communities', 'min'),
                    num_communities_max=('num_communities', 'max'),
                    community_size_mean=('mean_size', 'mean'),
                    community_size_min=('min_size', 'min'),
                    community_size_max=('max_size', 'max')
                )
                .reset_index()
            )

        meta = {
            'mode': mode,
            'algorithm': self.algorithm,
            'partition_type': partition_type_name,
            'weight_strategy': weight_strategy,
            'uniform_choice': uniform_choice,
            'uniform_scores': uniform_resolution_scores,
            'res_grid': res_grid
        }
        return meta, scans_df, best_df, summary_df

    def run_community_detection(
        self,
        graph_dict: Dict[int, Dict[int, nx.Graph]],
        use_optimization: bool = True,
        force_resolution: float | None = None,
    ) -> CommunityDetectionResult:
        """
        Run community detection over all graphs.

        Returns:
          partitions_df: per (subreddit_id, timestep) summary (+ community_sizes list).
          labels_array: {sub_id: {timestep: np.ndarray}} membership arrays (igraph order).
          labels_index_dict: {sub_id: {timestep: {node_idx: community_id}}}.
          labels_name_dict: {sub_id: {timestep: {node_name: community_id}}}.
          scans_df, best_df, summary_df: only when use_optimization=True.
          meta: algorithm + config + optimization metadata.
        """
        def _run_partition(ig_graph: ig.Graph, resolution: float | None):
            kwargs = {
                'weights': 'weight' if 'weight' in ig_graph.es.attributes() else None,
                'seed': self.seed
            }
            if resolution is not None and partition_cls is not la.ModularityVertexPartition:
                try:
                    return la.find_partition(ig_graph, partition_cls, resolution_parameter=resolution, **kwargs)
                except TypeError:
                    return la.find_partition(ig_graph, partition_cls, **kwargs)
            return la.find_partition(ig_graph, partition_cls, **kwargs)
        
        # 1. Decide per-graph resolutions
        optimization_meta = scans_df = best_df = summary_df = None
        chosen_res_map: Dict[tuple, float] = {}

        if use_optimization:
            optimization_meta, scans_df, best_df, summary_df = self.analyze_optimal_community_parameters(graph_dict)

            if best_df is not None and not best_df.empty:
                for _, row in best_df.iterrows():
                    chosen_res_map[(int(row.subreddit_id), int(row.timestep))] = float(row.resolution)
            else:
                if scans_df is not None and not scans_df.empty:
                    top_rows = (
                        scans_df.sort_values("metric_value", ascending=False)
                                .groupby(["subreddit_id", "timestep"])
                                .head(1)
                    )
                    for _, r in top_rows.iterrows():
                        chosen_res_map[(int(r.subreddit_id), int(r.timestep))] = float(r.resolution)
            
            # Default any missing (no constraints satisfied) to 1.0
            for sub_id, ts_dict in graph_dict.items():
                for ts in ts_dict.keys():
                    chosen_res_map.setdefault((sub_id, ts), 1.0)
        else:
            res_cfg = self.optimization_cfg.get('resolution', {})
            res_grid = res_cfg.get('grid', [1.0])
            default_res = res_grid[0]
            for sub_id, ts_dict in graph_dict.items():
                for ts in ts_dict.keys():
                    chosen_res_map[(sub_id, ts)] = default_res

        # Global override
        if force_resolution is not None:
            for k in chosen_res_map:
                chosen_res_map[k] = force_resolution

        # 2. Prepare partition configuration
        weight_strategy = self.community_configs.get('weights', {}).get('strategy', 'agreement_diff')
        algo_cfg = self.community_configs.get('algorithms', {}).get(self.algorithm, {})
        partition_type_name = algo_cfg.get('partition_type', 'RBConfigurationVertexPartition')
        partition_cls_map = {
            'RBConfigurationVertexPartition': la.RBConfigurationVertexPartition,
            'ModularityVertexPartition': la.ModularityVertexPartition,
            'CPMVertexPartition': la.CPMVertexPartition
        }
        partition_cls = partition_cls_map.get(partition_type_name, la.RBConfigurationVertexPartition)

        # 3. Execute partitions
        partition_rows = []
        
        labels_array_map: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)        # {sub_id: {ts: np.ndarray}}
        labels_dict_index_map: Dict[int, Dict[int, Dict[int, int]]] = defaultdict(dict)  # {sub_id: {ts: {node_idx: comm}}}
        labels_dict_name_map: Dict[int, Dict[int, Dict[str, int]]] = defaultdict(dict)   # {sub_id: {ts: {node_name: comm}}}

        for sub_id, ts_dict in graph_dict.items():
            for ts, G in ts_dict.items():
                if G.number_of_nodes() == 0:
                    continue
                resolution = chosen_res_map.get((sub_id, ts))
                ig_graph, node_to_idx, idx_to_node = self._convert_nx_to_igraph(G, weight_strategy=weight_strategy)
                partition = _run_partition(ig_graph, resolution)

                membership = partition.membership
                labels_arr = np.array(membership)
                idx_dict = {i: comm for i, comm in enumerate(membership)}
                name_dict = {idx_to_node[i]: comm for i, comm in enumerate(membership)}

                labels_array_map[sub_id][ts] = labels_arr
                labels_dict_index_map[sub_id][ts] = idx_dict
                labels_dict_name_map[sub_id][ts] = name_dict

                comm_sizes = np.bincount(membership).tolist()

                partition_rows.append({
                    'subreddit_id': sub_id,
                    'timestep': ts,
                    'resolution_used': resolution,
                    'num_nodes': G.number_of_nodes(),
                    'num_edges': G.number_of_edges(),
                    'num_communities': len(set(membership)),
                    'modularity': partition.modularity,
                    'community_sizes': comm_sizes
                })

        partitions_df = pd.DataFrame(partition_rows)

        result_meta = {
            'algorithm': self.algorithm,
            'partition_type': partition_type_name,
            'weight_strategy': weight_strategy,
            'use_optimization': use_optimization,
            'force_resolution': force_resolution,
            'optimization_meta': optimization_meta
        }

        return CommunityDetectionResult(
            meta=result_meta,
            partitions=partitions_df,
            labels_array=labels_array_map,
            labels_index_dict=labels_dict_index_map,
            labels_name_dict=labels_dict_name_map,
            optimization_scans=scans_df,
            optimization_best=best_df,
            optimization_summary=summary_df
        )