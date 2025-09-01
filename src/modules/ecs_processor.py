import numpy as np
import networkx as nx
import torch
import warnings
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from torch_geometric.utils import to_networkx
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

from src.baselines.echogae import EchoChamberMeasure, EchoGAE_algorithm
from src.modules.community_processor import LeidenCommunityProcessor
from src.utils.gnn_checkpointing import load_model_checkpoint


class ECSProcessor:
    """
    Complete Echo Chamber Score (ECS) Pipeline Processor
    
    Handles the complete ECS analysis pipeline including:
    - EchoGAE embedding generation
    - GNN embedding extraction
    - Community detection optimization
    - ECS calculation for both methods
    - Embedding-community alignment analysis
    - Results aggregation and analysis
    """
    
    def __init__(
        self,
        base_configs: Dict,
        community_configs: Dict,
        gnn_model_path: Optional[str] = None,
        gnn_model_class_path: Optional[str] = None,
        device: str = 'cpu',
        echogae_params: Optional[Dict] = None,
        verbose: bool = True
    ):
        """        
        Args:
            base_configs: Base configuration dictionary
            community_configs: Community detection configuration
            gnn_model_path: Path to trained GNN model checkpoint
            gnn_model_class_path: Import path for GNN model class
            device: Computing device ('cpu' or 'cuda')
            echogae_params: Parameters for EchoGAE algorithm
            verbose: Whether to print detailed progress
        """
        self.base_configs = base_configs
        self.community_configs = community_configs
        self.device = device
        self.verbose = verbose
        
        # EchoGAE default parameters
        self.echogae_params = {
            'show_progress': False,
            'epochs': 50,
            'hidden_channels': 100,
            'out_channels': 50,
            'seed': 42,
            **(echogae_params or {})
        }
        
        # Initialize community processor
        processed_path = base_configs.get('processed_path', 'data/processed')
        self.leiden_processor = LeidenCommunityProcessor(
            community_configs=community_configs, 
            output_dir=processed_path
        )
        
        # Load GNN model if provided
        self.gnn_model = None
        if gnn_model_path and gnn_model_class_path:
            self.gnn_model = self._load_gnn_model(gnn_model_path, gnn_model_class_path)
        
        # Suppress warnings
        self._suppress_warnings()
        
        # Results storage
        self.processed_dict = {}
    
    def _suppress_warnings(self):
        """Suppress common warnings during processing"""
        warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True to a scalar")
        warnings.filterwarnings("ignore", message="'train_test_split_edges' is deprecated")
    
    def _load_gnn_model(self, model_path: str, model_class_path: str):
        """Load pre-trained GNN model"""
        try:
            gnn_model, _ = load_model_checkpoint(
                model_path,
                device=self.device,
                model_class_path=model_class_path,
            )
            gnn_model.eval()
            if self.verbose:
                print(f"Loaded GNN model: {gnn_model.__class__.__name__}")
            return gnn_model
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load GNN model: {e}")
            return None
    
    def _extract_node_mapping(self, pyg_graph) -> Optional[List]:
        """Extract node index to user ID mapping from PyG graph"""
        if hasattr(pyg_graph, 'node_map') and pyg_graph.node_map is not None:
            # node_map is {user_id: index}, we need {index: user_id}
            user_to_index = pyg_graph.node_map
            index_to_user_mapping = {idx: user_id for user_id, idx in user_to_index.items()}
            
            # Convert to list ordered by index
            max_idx = max(index_to_user_mapping.keys()) if index_to_user_mapping else -1
            index_to_user_list = [None] * (max_idx + 1)
            for idx, user_id in index_to_user_mapping.items():
                index_to_user_list[idx] = user_id
            
            if self.verbose:
                print(f"  Extracted node mapping: {len(index_to_user_list)} nodes")
            
            return index_to_user_list
        
        if self.verbose:
            print("  No node_map found in PyG graph")
        return None
    
    def _validate_graph(self, nx_graph: nx.Graph, sub: int, ts: int) -> bool:
        """Validate graph has minimum requirements for processing"""
        if nx_graph.number_of_nodes() == 0:
            if self.verbose:
                print(f"  Skipping - empty graph for subreddit {sub}, timestep {ts}")
            return False
        
        # Get constraints from config
        min_communities_required = self.community_configs.get('optimization', {}).get('min_communities', 2)
        min_size_required = self.community_configs.get('optimization', {}).get('min_community_size', 10)
        theoretical_min_nodes = min_communities_required * min_size_required
        
        if nx_graph.number_of_nodes() < theoretical_min_nodes:
            if self.verbose:
                print(f"  WARNING: Graph has {nx_graph.number_of_nodes()} nodes but needs at least {theoretical_min_nodes} to satisfy constraints!")
        
        return True
    
    def _generate_echogae_embeddings(self, pyg_nx_graph: nx.Graph, node_features: np.ndarray) -> Optional[np.ndarray]:
        """Generate EchoGAE embeddings for the graph"""
        try:
            echogae_embeddings, _, _, _, _ = EchoGAE_algorithm(
                G=pyg_nx_graph.to_undirected(),
                user_embeddings=node_features,
                **self.echogae_params
            )
            return echogae_embeddings
        except Exception as e:
            if self.verbose:
                print(f"  Warning: EchoGAE failed: {e}")
            return None
    
    def _extract_gnn_embeddings(self, pyg_graph) -> Optional[np.ndarray]:
        """Extract GNN embeddings from PyG graph"""
        if self.gnn_model is None:
            return None
        
        try:
            gnn_embeddings = self.gnn_model.embed(pyg_graph, device=self.device, eval_mode=True)
            return gnn_embeddings.cpu().numpy() if gnn_embeddings is not None else None
        except Exception as e:
            if self.verbose:
                print(f"  Warning: GNN embedding extraction failed: {e}")
            return None
    
    def _detect_communities(self, graph_dict: Dict, sub: int, ts: int) -> Dict:
        """Run community detection for a single graph"""
        single_graph_dict = {sub: {ts: graph_dict[sub][ts]}}
        
        # Run optimization
        opt_meta, scans_df, best_df, summary_df = self.leiden_processor.analyze_optimal_community_parameters(single_graph_dict)
        
        # Run community detection with optimization
        community_result = self.leiden_processor.run_community_detection(
            graph_dict=single_graph_dict,
            use_optimization=True,
            save=False
        )
        
        return {
            'labels': community_result.labels_array[sub][ts],
            'partition': community_result.labels_name_dict[sub][ts],
            'modularity': self._extract_modularity(community_result, sub, ts)
        }
    
    def _extract_modularity(self, community_result, sub: int, ts: int) -> Optional[float]:
        """Extract modularity score from community detection results"""
        try:
            if hasattr(community_result, 'partitions') and not community_result.partitions.empty:
                partition_row = community_result.partitions[
                    (community_result.partitions['subreddit_id'] == sub) & 
                    (community_result.partitions['timestep'] == ts)
                ]
                if not partition_row.empty:
                    return partition_row['modularity'].iloc[0]
        except Exception:
            pass
        return None
    
    def _calculate_community_stats(self, comm_labels: np.ndarray, partition: Dict) -> Dict:
        """Calculate community statistics"""
        unique_communities = np.unique(comm_labels)
        comm_sizes = [np.sum(comm_labels == comm_id) for comm_id in unique_communities]
        
        # Build community nodes mapping
        comm_nodes = {comm_id: [] for comm_id in unique_communities}
        for user_id, comm_id in partition.items():
            if comm_id in comm_nodes:
                comm_nodes[comm_id].append(user_id)
        
        return {
            'num_communities': len(unique_communities),
            'comm_sizes': comm_sizes,
            'min_comm_size': min(comm_sizes) if comm_sizes else 0,
            'max_comm_size': max(comm_sizes) if comm_sizes else 0,
            'mean_comm_size': np.mean(comm_sizes) if comm_sizes else 0,
            'comm_nodes': comm_nodes,
            'unique_communities': unique_communities
        }
    
    def _analyze_embedding_community_alignment(self, embeddings: np.ndarray, comm_labels: np.ndarray, 
                                             comm_nodes: Dict, method_name: str) -> Dict:
        """Check how well embeddings align with community structure"""
        try:
            # Calculate silhouette score (higher = better cluster separation)
            sil_score = silhouette_score(embeddings, comm_labels)
            
            # Calculate average intra/inter community distances
            dist_matrix = squareform(pdist(embeddings, metric='cosine'))
            
            intra_distances = []
            inter_distances = []
            
            for i in range(len(comm_labels)):
                for j in range(i+1, len(comm_labels)):
                    if comm_labels[i] == comm_labels[j]:
                        intra_distances.append(dist_matrix[i, j])
                    else:
                        inter_distances.append(dist_matrix[i, j])
            
            avg_intra = np.mean(intra_distances) if intra_distances else 0
            avg_inter = np.mean(inter_distances) if inter_distances else 0
            
            if self.verbose:
                print(f"    {method_name} Embedding-Community Alignment:")
                print(f"    Silhouette Score: {sil_score:.4f} (higher = better separation)")
                print(f"    + Avg Intra-Community Distance: {avg_intra:.4f}")
                print(f"    + Avg Inter-Community Distance: {avg_inter:.4f}")
                print(f"    + Distance Ratio (Inter/Intra): {avg_inter/avg_intra:.4f}" if avg_intra > 0 else "    + Distance Ratio: inf")

            return {
                'silhouette': sil_score,
                'avg_intra_dist': avg_intra,
                'avg_inter_dist': avg_inter,
                'distance_ratio': avg_inter/avg_intra if avg_intra > 0 else float('inf')
            }
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Alignment analysis failed for {method_name}: {e}")
            return {
                'silhouette': 0.0,
                'avg_intra_dist': 0.0,
                'avg_inter_dist': 0.0,
                'distance_ratio': float('inf')
            }
    
    def _calculate_ecs_scores(self, embeddings: np.ndarray, comm_labels: np.ndarray) -> Tuple[float, List[float]]:
        """Calculate Echo Chamber Scores for embeddings"""
        try:
            ecm = EchoChamberMeasure(
                users_representations=embeddings,
                labels=comm_labels,
            )
            overall_eci = ecm.echo_chamber_index()
            community_ecis = [ecm.community_echo_chamber_index(i) for i in np.unique(comm_labels)]
            return overall_eci, community_ecis
        except Exception as e:
            if self.verbose:
                print(f"  Warning: ECS calculation failed: {e}")
            return 0.0, []
    
    def _process_single_graph(self, pyg_graph, graph_dict: Dict) -> Optional[Dict]:
        """Process a single PyG graph through the complete ECS pipeline"""
        sub = getattr(pyg_graph, 'subreddit_id', None)
        ts = getattr(pyg_graph, 'local_timestep', None)
        
        if sub is None or ts is None:
            if self.verbose:
                print("Skipping - missing subreddit_id or timestep")
            return None
        
        if self.verbose:
            print(f"Processing graph for subreddit: {sub}, timestep: {ts}")
        
        # Extract node mapping
        index_to_user_mapping = self._extract_node_mapping(pyg_graph)
        
        # Extract node features
        node_features = pyg_graph.x.cpu().numpy() if hasattr(pyg_graph, 'x') and pyg_graph.x is not None else None
        
        # Use original graph with node names from graph_dict
        nx_graph = graph_dict[sub][ts]
        
        # Convert PyG to nx for EchoGAE
        pyg_nx_graph = to_networkx(pyg_graph, to_undirected=False)
        
        # Validate graph
        if not self._validate_graph(nx_graph, sub, ts):
            return None
        
        if self.verbose:
            print(f"  Graph stats: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
        
        # Generate embeddings
        echogae_embeddings = self._generate_echogae_embeddings(pyg_nx_graph, node_features) if node_features is not None else None
        gnn_embeddings = self._extract_gnn_embeddings(pyg_graph)
        
        # Detect communities
        community_info = self._detect_communities(graph_dict, sub, ts)
        comm_labels = community_info['labels']
        partition = community_info['partition']
        modularity = community_info['modularity']
        
        # Calculate community statistics
        comm_stats = self._calculate_community_stats(comm_labels, partition)
        
        if self.verbose:
            print(f"  Communities found: {comm_stats['num_communities']}")
            print(f"  Community sizes (min={comm_stats['min_comm_size']}): {comm_stats['comm_sizes']}")
            print(f"  Modularity: {modularity:.3f}" if modularity else "  Modularity: None")
        
        # Calculate ECS scores
        echogae_eci, echogae_comm_eci = self._calculate_ecs_scores(echogae_embeddings, comm_labels) if echogae_embeddings is not None else (0.0, [])
        debgnn_eci, debgnn_comm_eci = self._calculate_ecs_scores(gnn_embeddings, comm_labels) if gnn_embeddings is not None else (0.0, [])
        
        if self.verbose:
            print(f"  EchoGAE ECI: {echogae_eci:.4f} | Community ECIs: {[f'{eci:.4f}' for eci in echogae_comm_eci]}")
            print(f"  DebateGNN ECI: {debgnn_eci:.4f} | Community ECIs: {[f'{eci:.4f}' for eci in debgnn_comm_eci]}")
        
        # Analyze embedding-community alignment
        if self.verbose:
            print(f"  **Embedding-Community Alignment Analysis**")

        echo_alignment = self._analyze_embedding_community_alignment(
            echogae_embeddings, comm_labels, comm_stats['comm_nodes'], "EchoGAE"
        ) if echogae_embeddings is not None else {}
        
        debgnn_alignment = self._analyze_embedding_community_alignment(
            gnn_embeddings, comm_labels, comm_stats['comm_nodes'], "DebateGNN"
        ) if gnn_embeddings is not None else {}
        
        if self.verbose:
            print(f"  --- End Alignment Analysis ---\n")
        
        # Build comprehensive community info
        subreddit_map = {v: k for k, v in self.base_configs['subreddits'].items()}
        detailed_community_info = {
            'subreddit_id': sub,
            'subreddit': subreddit_map.get(sub, f'Unknown_{sub}'),
            'timestep': ts,
            'num_nodes': nx_graph.number_of_nodes(),
            'num_edges': nx_graph.number_of_edges(),
            'modularity': modularity,
            'partition': partition,
            'comm_labels': comm_labels,
            **comm_stats
        }
        
        return {
            'echogae_embeddings': echogae_embeddings,
            'gnn_embeddings': gnn_embeddings,
            'node_features': node_features,
            'community_info': detailed_community_info,
            'echogae_eci': echogae_eci,
            'echogae_comm_eci': echogae_comm_eci,
            'debgnn_eci': debgnn_eci,
            'debgnn_comm_eci': debgnn_comm_eci,
            'nx_graph': nx_graph,
            'echogae_alignment': echo_alignment,
            'debgnn_alignment': debgnn_alignment,
            'index_to_user': index_to_user_mapping
        }
    
    def process_graphs(self, pyg_graphs: List, graph_dict: Dict) -> Dict:
        """
        Process multiple PyG graphs through the ECS pipeline
        
        Args:
            pyg_graphs: List of PyTorch Geometric graphs
            graph_dict: Dictionary containing graph data by subreddit and timestep
            
        Returns:
            processed_dict: Dictionary containing all processing results
        """
        self.processed_dict = {}
        
        if self.verbose:
            print(f"Starting ECS processing for {len(pyg_graphs)} graphs")
            print("=" * 80)
        
        for pyg_graph in pyg_graphs:
            result = self._process_single_graph(pyg_graph, graph_dict)
            
            if result is not None:
                sub = result['community_info']['subreddit_id']
                ts = result['community_info']['timestep']
                
                if sub not in self.processed_dict:
                    self.processed_dict[sub] = {}
                
                self.processed_dict[sub][ts] = result
        
        if self.verbose:
            print(f"\nProcessed {len(self.processed_dict)} subreddits")
            print("=" * 60)
        
        return self.processed_dict
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Convert processed results to a pandas DataFrame
        
        Returns:
            DataFrame with ECS results and metadata
        """
        combined_data = []
        
        for subreddit_id, timesteps in self.processed_dict.items():
            for timestep, data in timesteps.items():
                community_info = data['community_info']
                
                row = {
                    'subreddit_id': subreddit_id,
                    'subreddit': community_info['subreddit'],
                    'timestep': timestep,
                    'num_nodes': community_info['num_nodes'],
                    'num_edges': community_info['num_edges'],
                    'num_communities': community_info['num_communities'],
                    'modularity': round(community_info['modularity'], 3) if community_info['modularity'] is not None else None,
                    'min_comm_size': community_info['min_comm_size'],
                    'max_comm_size': community_info['max_comm_size'],
                    'mean_comm_size': round(community_info['mean_comm_size'], 1),
                    'echogae_eci': round(data['echogae_eci'], 3),
                    'debgnn_eci': round(data['debgnn_eci'], 3),
                    # Convert numpy arrays to Python lists with native types
                    'comm_sizes': [int(size) for size in community_info['comm_sizes']],
                    'echogae_comm_eci': [round(float(eci), 3) for eci in data['echogae_comm_eci']],
                    'debgnn_comm_eci': [round(float(eci), 3) for eci in data['debgnn_comm_eci']],
                    'echogae_silhouette': round(data['echogae_alignment'].get('silhouette', 0), 4),
                    'debgnn_silhouette': round(data['debgnn_alignment'].get('silhouette', 0), 4),
                }
                
                combined_data.append(row)
        
        ecs_df = pd.DataFrame(combined_data)
        ecs_df = ecs_df.sort_values(['subreddit_id', 'timestep']).reset_index(drop=True)
        
        # Calculate deltas between timesteps
        ecs_df['delta_echogae_eci'] = ecs_df.groupby('subreddit_id')['echogae_eci'].diff().round(4)
        ecs_df['delta_debgnn_eci'] = ecs_df.groupby('subreddit_id')['debgnn_eci'].diff().round(4)
        
        return ecs_df
    
    def save_results(self, output_path: str = 'results/ecs_results.csv') -> pd.DataFrame:
        """Save results to CSV file"""
        ecs_df = self.get_results_dataframe()
        ecs_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"Results saved to {output_path}")
        
        return ecs_df
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics grouped by subreddit"""
        ecs_df = self.get_results_dataframe()
        
        summary = ecs_df.groupby('subreddit').agg({
            'timestep': 'count',
            'echogae_eci': ['mean', 'min', 'max', 'std'],
            'delta_echogae_eci': ['mean', 'min', 'max', 'std'],
            'debgnn_eci': ['mean', 'min', 'max', 'std'],
            'delta_debgnn_eci': ['mean', 'min', 'max', 'std'],
            'echogae_silhouette': ['mean', 'min', 'max', 'std'],
            'debgnn_silhouette': ['mean', 'min', 'max', 'std'],
        }).round(4)
        
        return summary
    
    def get_processed_data(self) -> Dict:
        """Get the complete processed data dictionary"""
        return self.processed_dict


# Usage example
def create_ecs_processor(base_configs, community_configs, 
                        gnn_model_path="checkpoints/best_model_2508152029.pth",
                        gnn_model_class_path="src.models.multitask_debate_gnn.MultitaskDebateGNN",
                        device='cuda'):
    """Factory function to create ECS processor with common defaults"""
    
    return ECSProcessor(
        base_configs=base_configs,
        community_configs=community_configs,
        gnn_model_path=gnn_model_path,
        gnn_model_class_path=gnn_model_class_path,
        device=device,
        echogae_params={
            'show_progress': False,
            'epochs': 50,
            'hidden_channels': 100,
            'out_channels': 50,
            'seed': 42
        },
        verbose=True
    )