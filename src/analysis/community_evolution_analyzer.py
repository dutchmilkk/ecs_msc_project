import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional


class CommunityEvolutionAnalyzer:
    """Analyze community evolution using Jaccard similarity and Hungarian algorithm matching."""
    
    def __init__(self, verbose: bool = True, matrix_display_limit: Optional[int] = None):
        """
        Initialize the Community Evolution Analyzer.
        
        Args:
            verbose: Whether to print detailed output during analysis
            matrix_display_limit: Maximum number of rows/columns to display in matrix 
                                 (None = show all, int = limit)
        """
        self.verbose = verbose
        self.matrix_display_limit = matrix_display_limit
    
    @staticmethod
    def compute_jaccard_similarity(c1: List, c2: List) -> float:
        """Compute Jaccard similarity between two communities (sets of users)."""
        set1, set2 = set(c1), set(c2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def compute_community_jaccard_matrix(self, comms_t1: Dict, comms_t2: Dict) -> Tuple[np.ndarray, List, List]:
        """
        Compute Jaccard similarity matrix between communities at two timesteps.
        
        Args:
            comms_t1: dict {community_id: list_of_users} for timestep t1
            comms_t2: dict {community_id: list_of_users} for timestep t2
        
        Returns:
            jaccard_matrix: 2D numpy array where [i,j] = Jaccard(comm_i_t1, comm_j_t2)
            comm_ids_t1: list of community IDs for t1 (rows)
            comm_ids_t2: list of community IDs for t2 (columns)
        """
        comm_ids_t1 = sorted(comms_t1.keys())
        comm_ids_t2 = sorted(comms_t2.keys())
        
        n_t1, n_t2 = len(comm_ids_t1), len(comm_ids_t2)
        jaccard_matrix = np.zeros((n_t1, n_t2))
        
        for i, comm_id_t1 in enumerate(comm_ids_t1):
            for j, comm_id_t2 in enumerate(comm_ids_t2):
                jaccard_matrix[i, j] = self.compute_jaccard_similarity(
                    comms_t1[comm_id_t1], 
                    comms_t2[comm_id_t2]
                )
        
        return jaccard_matrix, comm_ids_t1, comm_ids_t2
    
    def find_best_matches_hungarian(self, jaccard_matrix: np.ndarray, comm_ids_t1: List, 
                                   comm_ids_t2: List, comms_t1: Dict, comms_t2: Dict) -> Tuple[List, float]:
        """
        Find optimal community matches using Hungarian algorithm.
        
        Args:
            jaccard_matrix: Jaccard similarity matrix
            comm_ids_t1, comm_ids_t2: Community IDs for each timestep
            comms_t1, comms_t2: Community user dictionaries
        
        Returns:
            matches: List of match dictionaries with optimal assignments
            total_jaccard: Sum of Jaccard similarities for optimal matching
        """
        # Hungarian algorithm minimizes cost, but we want to maximize Jaccard
        cost_matrix = 1 - jaccard_matrix
        
        # Handle case where dimensions don't match by padding with high cost
        n_t1, n_t2 = jaccard_matrix.shape
        if n_t1 != n_t2:
            max_dim = max(n_t1, n_t2)
            padded_cost_matrix = np.ones((max_dim, max_dim))
            padded_cost_matrix[:n_t1, :n_t2] = cost_matrix
            cost_matrix = padded_cost_matrix
        
        # Run Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid matches (not dummy assignments)
        matches = []
        total_jaccard = 0.0
        
        for i, j in zip(row_indices, col_indices):
            if i < n_t1 and j < n_t2:  # Valid assignment (not dummy)
                comm_t1 = comm_ids_t1[i]
                comm_t2 = comm_ids_t2[j]
                jaccard_score = jaccard_matrix[i, j]
                
                matches.append({
                    't1_comm': comm_t1,
                    't2_comm': comm_t2,
                    'jaccard': jaccard_score,
                    't1_size': len(comms_t1[comm_t1]),
                    't2_size': len(comms_t2[comm_t2]),
                    'assignment_type': 'matched'
                })
                total_jaccard += jaccard_score
        
        # Handle unmatched communities
        matched_t1 = {match['t1_comm'] for match in matches}
        matched_t2 = {match['t2_comm'] for match in matches}
        
        # Unmatched t1 communities (disappeared)
        for comm_t1 in comm_ids_t1:
            if comm_t1 not in matched_t1:
                matches.append({
                    't1_comm': comm_t1,
                    't2_comm': None,
                    'jaccard': 0.0,
                    't1_size': len(comms_t1[comm_t1]),
                    't2_size': 0,
                    'assignment_type': 'disappeared'
                })
        
        # Unmatched t2 communities (emerged)
        for comm_t2 in comm_ids_t2:
            if comm_t2 not in matched_t2:
                matches.append({
                    't1_comm': None,
                    't2_comm': comm_t2,
                    'jaccard': 0.0,
                    't1_size': 0,
                    't2_size': len(comms_t2[comm_t2]),
                    'assignment_type': 'emerged'
                })
        
        return matches, float(total_jaccard)
    
    @staticmethod
    def analyze_user_migration_statistics(comms_t1: Dict, comms_t2: Dict) -> Dict:
        """Analyze user migration patterns between timesteps."""
        users_t1 = set().union(*comms_t1.values()) if comms_t1 else set()
        users_t2 = set().union(*comms_t2.values()) if comms_t2 else set()
        
        retained_users = users_t1.intersection(users_t2)
        new_users = users_t2 - users_t1
        lost_users = users_t1 - users_t2
        
        return {
            'total_t1': len(users_t1),
            'total_t2': len(users_t2),
            'retained': len(retained_users),
            'new': len(new_users),
            'lost': len(lost_users),
            'retention_rate': len(retained_users) / len(users_t1) if users_t1 else 0,
            'growth_rate': len(new_users) / len(users_t1) if users_t1 else 0
        }
    
    def analyze_evolution(self, subreddit_data: Dict) -> Optional[Dict]:
        """
        Compute Jaccard evolution for a single subreddit across timesteps using Hungarian algorithm.
        
        Args:
            subreddit_data: dict {timestep: {'community_info': {...}, ...}}
        
        Returns:
            evolution_data: dict containing evolution analysis results
        """
        ts = sorted(subreddit_data.keys())
        if len(ts) < 2:
            if self.verbose:
                print("Not enough timesteps to compute evolution.")
            return None
        
        evolution_data = {
            'timesteps': ts,
            'ts_pairs': [],
            'jaccard_matrices': [],
            'hungarian_matches': [],
            'total_jaccards': [],
            'migration_stats': [],
            'comm_ids_t1': [],
            'comm_ids_t2': []
        }
        
        if self.verbose:
            subreddit_name = subreddit_data[ts[0]]['community_info']['subreddit']
            print(f"Computing evolution for {subreddit_name} across {len(ts)} timesteps: {ts}")
        
        for i in range(len(ts) - 1):
            t1, t2 = ts[i], ts[i + 1]
            comm_data_t1 = subreddit_data[t1]['community_info']
            comm_data_t2 = subreddit_data[t2]['community_info']
            
            # Extract community nodes
            comms_t1 = comm_data_t1['comm_nodes']  # {comm_id: [users]}
            comms_t2 = comm_data_t2['comm_nodes']  # {comm_id: [users]}
            
            # Compute Jaccard matrix
            jaccard_matrix, comm_ids_t1, comm_ids_t2 = self.compute_community_jaccard_matrix(comms_t1, comms_t2)
            
            # Find optimal matches using Hungarian algorithm
            hungarian_matches, total_jaccard = self.find_best_matches_hungarian(
                jaccard_matrix, comm_ids_t1, comm_ids_t2, comms_t1, comms_t2
            )
            
            # Compute migration statistics
            migration_stats = self.analyze_user_migration_statistics(comms_t1, comms_t2)
            
            # Store results
            evolution_data['ts_pairs'].append((t1, t2))
            evolution_data['jaccard_matrices'].append(jaccard_matrix)
            evolution_data['hungarian_matches'].append(hungarian_matches)
            evolution_data['total_jaccards'].append(total_jaccard)
            evolution_data['migration_stats'].append(migration_stats)
            evolution_data['comm_ids_t1'].append(comm_ids_t1)
            evolution_data['comm_ids_t2'].append(comm_ids_t2)
            
            if self.verbose:
                print(f"\n--- Timestep {t1} -> {t2} ---")
                print(f"Communities: {len(comms_t1)} -> {len(comms_t2)}")
                print(f"Total Jaccard (Hungarian): {total_jaccard:.3f}")
                print(f"User Retention: {migration_stats['retention_rate']:.1%}")
        
        return evolution_data
    
    def analyze_multiple_subreddits(self, processed_dict: Dict) -> Dict:
        """
        Analyze evolution for multiple subreddits.
        
        Args:
            processed_dict: dict {subreddit_id: {timestep: data}}
        
        Returns:
            all_evolution_data: dict {subreddit_id: evolution_data}
        """
        all_evolution_data = {}
        
        for subreddit_id, subreddit_data in processed_dict.items():
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"ANALYZING SUBREDDIT {subreddit_id}")
                print(f"{'='*60}")
            
            evolution_data = self.analyze_evolution(subreddit_data)
            if evolution_data is not None:
                all_evolution_data[subreddit_id] = evolution_data
        
        return all_evolution_data