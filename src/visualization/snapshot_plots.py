import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, Optional, List

from src.utils.community_colors import build_lineage_colors, build_unique_node_colors

def plot_snapshot_analysis(processed_dict_single_subreddit, timestep, evolution_data=None, 
                          subreddit_name=None, figsize=(20, 8), 
                          color_mode="lineage", draw_boundaries=True):
    """
    Create 3 subplots: EchoGAE UMAP, DebateGNN UMAP, and NetworkX graph visualization
    
    Args:
        processed_dict_single_subreddit: dict {timestep: data} for a single subreddit
        timestep: int, timestep to analyze
        evolution_data: optional evolution data for lineage colors
        subreddit_name: optional name for the subreddit
        figsize: tuple for figure size
        color_mode: str, "lineage" (uses evolution_data) or "unique" (unique per community) or "default" (simple cycling)
        draw_boundaries: bool, whether to draw community boundary lines on NetworkX graph
    """
    if timestep not in processed_dict_single_subreddit:
        print(f"Timestep {timestep} not found in data")
        return
    
    # Import color utilities (assuming they exist)
    try:
        from src.utils.community_colors import build_lineage_colors, build_unique_node_colors
    except ImportError:
        print("Color utilities not found, using basic color scheme")
        build_lineage_colors = None
        build_unique_node_colors = None
    
    data = processed_dict_single_subreddit[timestep]
    community_info = data['community_info']
    
    # Extract data
    echogae_embeddings = data['echogae_embeddings']
    gnn_embeddings = data['gnn_embeddings']
    nx_graph = data['nx_graph']
    comm_labels = community_info['comm_labels']
    comm_nodes = community_info['comm_nodes']
    partition = community_info['partition']  # user_id -> comm_id mapping
    
    # Check overlap between nx nodes and partition keys
    nx_nodes_set = set(nx_graph.nodes())
    partition_keys_set = set(partition.keys())
    overlap = nx_nodes_set.intersection(partition_keys_set)
    print(f"  Overlap between nx_nodes and partition keys: {len(overlap)}/{len(nx_nodes_set)}")
    
    # CREATE COLOR MAP based on color_mode
    unique_communities = sorted(comm_nodes.keys())
    
    if color_mode == "lineage" and evolution_data is not None and build_lineage_colors is not None:
        print("  Using lineage colors consistent with Sankey diagrams")
        node_color_map = build_lineage_colors(
            processed_dict_single_subreddit,
            evolution_data,
            mode="hungarian",
            min_jaccard_parent=0.05,
            split_strategy="new_hues"
        )
        # Create community color mapping using lineage colors
        comm_color_map = {}
        for comm_id in unique_communities:
            node_key = f"{timestep}_{comm_id}"
            hex_color = node_color_map.get(node_key, "#999999")
            comm_color_map[comm_id] = hex_color
        color_info = "Lineage Colors"
        
    elif color_mode == "unique" and build_unique_node_colors is not None:
        print("  Using unique colors per community")
        node_color_map = build_unique_node_colors(processed_dict_single_subreddit)
        # Create community color mapping using unique colors
        comm_color_map = {}
        for comm_id in unique_communities:
            node_key = f"{timestep}_{comm_id}"
            hex_color = node_color_map.get(node_key, "#999999")
            comm_color_map[comm_id] = hex_color
        color_info = "Unique Colors"
        
    else:  # color_mode == "default" or fallback
        print("  Using default color cycling")
        # Simple color cycling using matplotlib default colors
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        comm_color_map = {}
        for i, comm_id in enumerate(unique_communities):
            comm_color_map[comm_id] = default_colors[i % len(default_colors)]
        color_info = "Default Colors"
    
    print(f"  Community color map: {comm_color_map}")
    
    # Create node colors array based on community labels (same order as embeddings)
    node_colors = [comm_color_map[comm_labels[i]] for i in range(len(comm_labels))]
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    title_suffix = f" - {subreddit_name} T{timestep}" if subreddit_name else f" - T{timestep}"
    fig.suptitle(f'Snapshot Analysis{title_suffix}', fontsize=16, fontweight='bold')
    
    # Plot 1: EchoGAE UMAP
    if echogae_embeddings is not None and echogae_embeddings.shape[0] > 0:
        try:
            import umap
            import umap.plot
            
            # Fit UMAP reducer
            reducer_echo = umap.UMAP(
                n_components=2, 
                random_state=42, 
                n_neighbors=min(15, echogae_embeddings.shape[0]-1)
            )
            echo_embedding = reducer_echo.fit_transform(echogae_embeddings)
            # Ensure echo_embedding is a numpy array
            echo_embedding = np.array(echo_embedding)
            
            # Create interactive UMAP scatter plot with Plotly
            node_names = list(partition.keys()) if partition else list(nx_graph.nodes()) if nx_graph.number_of_nodes() > 0 else [f"Node_{i}" for i in range(len(comm_labels))]
            
            # Prepare hover data
            hover_data = []
            for i in range(len(comm_labels)):
                comm_id = comm_labels[i]
                comm_size = len(comm_nodes[comm_id])
                echo_eci = data['echogae_comm_eci'][list(unique_communities).index(comm_id)]
                node_name = node_names[i] if i < len(node_names) else f"Node_{i}"
                
                hover_data.append({
                    'Node_Name': str(node_name),
                    'Node_Index': i,
                    'Community_ID': comm_id,
                    'Community_Size': comm_size,
                    'Community_ECI': f"{echo_eci:.4f}",
                    'X_Coord': f"{echo_embedding[i, 0]:.3f}",
                    'Y_Coord': f"{echo_embedding[i, 1]:.3f}"
                })
            
            # Create interactive Plotly scatter plot
            fig_echo_interactive = go.Figure()
            
            # Add scatter points grouped by community for better legend
            for comm_id in unique_communities:
                comm_mask = comm_labels == comm_id
                comm_indices = np.where(comm_mask)[0]
                
                fig_echo_interactive.add_trace(go.Scatter(
                    x=echo_embedding[comm_mask, 0],
                    y=echo_embedding[comm_mask, 1],
                    mode='markers',
                    marker=dict(
                        color=comm_color_map[comm_id],
                        size=8,
                        line=dict(color='black', width=0.5)
                    ),
                    name=f'Community {comm_id} ({len(comm_indices)})',
                    hovertemplate='<b>%{customdata[0]}</b><br>' +
                                 'Node Index: %{customdata[1]}<br>' +
                                 'Community: %{customdata[2]}<br>' +
                                 'Comm Size: %{customdata[3]}<br>' +
                                 'Comm ECI: %{customdata[4]}<br>' +
                                 'UMAP X: %{customdata[5]}<br>' +
                                 'UMAP Y: %{customdata[6]}<br>' +
                                 '<extra></extra>',
                    customdata=[[hover_data[i]['Node_Name'], 
                               hover_data[i]['Node_Index'],
                               hover_data[i]['Community_ID'],
                               hover_data[i]['Community_Size'],
                               hover_data[i]['Community_ECI'],
                               hover_data[i]['X_Coord'],
                               hover_data[i]['Y_Coord']] for i in comm_indices]
                ))
            
            fig_echo_interactive.update_layout(
                title=f'EchoGAE Embeddings (UMAP) - r/{subreddit_name}, ts:{timestep}<br>ECI: {data["echogae_eci"]:.4f}',
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                width=600,
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.01
                )
            )
            
            fig_echo_interactive.show()
            
            # Still create the regular matplotlib scatter plot for the subplot
            scatter1 = ax1.scatter(echo_embedding[:, 0], echo_embedding[:, 1], 
                                 c=node_colors, s=80, alpha=0.7, 
                                 edgecolors='black', linewidth=0.5)
            ax1.set_title(f'EchoGAE Embeddings (UMAP)\nECI: {data["echogae_eci"]:.4f}')
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            ax1.grid(True, alpha=0.3)
            
        except ImportError as e:
            print(f"UMAP not available: {e}")
            ax1.text(0.5, 0.5, 'UMAP not available\nInstall: pip install umap-learn', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('EchoGAE Embeddings (UMAP)')
    else:
        ax1.text(0.5, 0.5, 'No EchoGAE embeddings', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('EchoGAE Embeddings (UMAP)')

    # Plot 2: DebateGNN UMAP
    if gnn_embeddings is not None and gnn_embeddings.shape[0] > 0:
        try:
            import umap
            import umap.plot
            
            reducer_gnn = umap.UMAP(
                n_components=2, 
                random_state=42,
                n_neighbors=min(15, gnn_embeddings.shape[0]-1)
            )
            gnn_embedding = reducer_gnn.fit_transform(gnn_embeddings)
            # Ensure gnn_embedding is a numpy array for proper indexing
            gnn_embedding = np.array(gnn_embedding)
            
            # Create interactive UMAP scatter plot for DebateGNN
            hover_data_gnn = []
            for i in range(len(comm_labels)):
                comm_id = comm_labels[i]
                comm_size = len(comm_nodes[comm_id])
                debgnn_eci = data['debgnn_comm_eci'][list(unique_communities).index(comm_id)]
                node_name = node_names[i] if i < len(node_names) else f"Node_{i}"
                
                hover_data_gnn.append({
                    'Node_Name': str(node_name),
                    'Node_Index': i,
                    'Community_ID': comm_id,
                    'Community_Size': comm_size,
                    'Community_ECI': f"{debgnn_eci:.4f}",
                    'X_Coord': f"{gnn_embedding[i, 0]:.3f}",
                    'Y_Coord': f"{gnn_embedding[i, 1]:.3f}"
                })
            
            # Create interactive Plotly scatter plot for DebateGNN
            fig_gnn_interactive = go.Figure()
            
            for comm_id in unique_communities:
                comm_mask = comm_labels == comm_id
                comm_indices = np.where(comm_mask)[0]
                
                fig_gnn_interactive.add_trace(go.Scatter(
                    x=gnn_embedding[comm_mask, 0],
                    y=gnn_embedding[comm_mask, 1],
                    mode='markers',
                    marker=dict(
                        color=comm_color_map[comm_id],
                        size=8,
                        line=dict(color='black', width=0.5)
                    ),
                    name=f'Community {comm_id} ({len(comm_indices)})',
                    hovertemplate='<b>%{customdata[0]}</b><br>' +
                                 'Node Index: %{customdata[1]}<br>' +
                                 'Community: %{customdata[2]}<br>' +
                                 'Comm Size: %{customdata[3]}<br>' +
                                 'Comm ECI: %{customdata[4]}<br>' +
                                 'UMAP X: %{customdata[5]}<br>' +
                                 'UMAP Y: %{customdata[6]}<br>' +
                                 '<extra></extra>',
                    customdata=[[hover_data_gnn[i]['Node_Name'], 
                               hover_data_gnn[i]['Node_Index'],
                               hover_data_gnn[i]['Community_ID'],
                               hover_data_gnn[i]['Community_Size'],
                               hover_data_gnn[i]['Community_ECI'],
                               hover_data_gnn[i]['X_Coord'],
                               hover_data_gnn[i]['Y_Coord']] for i in comm_indices]
                ))
            
            fig_gnn_interactive.update_layout(
                title=f'DebateGNN Embeddings (UMAP) - r/{subreddit_name}, ts:{timestep}<br>ECI: {data["debgnn_eci"]:.4f}',
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                width=600,
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.01
                )
            )
            
            fig_gnn_interactive.show()
            
            # Regular matplotlib scatter plot
            scatter2 = ax2.scatter(gnn_embedding[:, 0], gnn_embedding[:, 1], 
                                 c=node_colors, s=80, alpha=0.7, 
                                 edgecolors='black', linewidth=0.5)
            ax2.set_title(f'DebateGNN Embeddings (UMAP)\nECI: {data["debgnn_eci"]:.4f}')
            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
            ax2.grid(True, alpha=0.3)
            
        except ImportError as e:
            print(f"UMAP not available: {e}")
            ax2.text(0.5, 0.5, 'UMAP not available\nInstall: pip install umap-learn', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('DebateGNN Embeddings (UMAP)')
    else:
        ax2.text(0.5, 0.5, 'No DebateGNN embeddings', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('DebateGNN Embeddings (UMAP)')
    
    # Plot 3: NetworkX Graph
    if nx_graph.number_of_nodes() > 0:
        # Create node color mapping for NetworkX (map user_ids to colors)
        nx_node_colors = []
        nodes_with_colors = 0
        nodes_without_colors = 0
        
        for node in nx_graph.nodes():
            if node in partition:
                comm_id = partition[node]
                color = comm_color_map.get(comm_id, 'gray')
                nx_node_colors.append(color)
                nodes_with_colors += 1
            else:
                nx_node_colors.append('gray')
                nodes_without_colors += 1
        
        print(f"  NetworkX coloring: {nodes_with_colors} nodes with colors, {nodes_without_colors} nodes without colors")
        
        # Use community-aware layout for positioning
        if nx_graph.number_of_nodes() <= 500:  # Only for smaller graphs
            try:
                # Community-aware layout
                communities = {}
                for node, comm_id in partition.items():
                    if node in nx_graph.nodes():  # Only include nodes that exist in graph
                        if comm_id not in communities:
                            communities[comm_id] = set()
                        communities[comm_id].add(node)
                
                # Convert to list of sets
                community_list = [communities[comm_id] for comm_id in sorted(communities.keys())]
                
                print(f"  Community layout: {len(community_list)} communities for positioning")
                
                # Create initial positions where communities are separated
                pos = {}
                import math
                n_communities = len(community_list)
                
                if n_communities > 1:
                    # Arrange communities in a circle
                    for i, community in enumerate(community_list):
                        angle = 2 * math.pi * i / n_communities
                        center_x = 3 * math.cos(angle)  # Communities spread in circle
                        center_y = 3 * math.sin(angle)
                        
                        # Position nodes within each community using spring layout
                        if len(community) > 1:
                            subgraph = nx_graph.subgraph(community)
                            sub_pos = nx.spring_layout(subgraph, k=0.3, iterations=30)
                            # Offset the subgraph positions to the community center
                            for node, (x, y) in sub_pos.items():
                                pos[node] = (center_x + x, center_y + y)
                        else:
                            # Single node community
                            node = list(community)[0]
                            pos[node] = (center_x, center_y)
                else:
                    # Fallback to regular spring layout if only one community
                    pos = nx.spring_layout(nx_graph, k=1/np.sqrt(nx_graph.number_of_nodes()), iterations=50, seed=42)
                
                # Refine positions with spring layout using initial positions
                pos = nx.spring_layout(nx_graph, pos=pos, k=1.0, iterations=50, seed=42)
                
            except Exception as e:
                print(f"  Community layout failed ({e}), using regular spring layout")
                # Fallback to regular spring layout
                pos = nx.spring_layout(nx_graph, k=1/np.sqrt(nx_graph.number_of_nodes()), iterations=50, seed=42)
            
            # Draw the graph
            nx.draw_networkx_nodes(nx_graph, pos, node_color=nx_node_colors, node_size=80,
                                alpha=0.8, ax=ax3, edgecolors='black', linewidths=0.5)
            nx.draw_networkx_edges(nx_graph, pos, alpha=0.4, width=0.8, ax=ax3, arrows=True, 
                                arrowsize=8, edge_color='#303030', 
                                min_source_margin=0, min_target_margin=0, connectionstyle="arc3,rad=0.05")

                    
            # Optional: Add community boundary circles
            if draw_boundaries and len(community_list) > 1 and nx_graph.number_of_nodes() <= 200:
                print(f"  Drawing community boundaries for {len(community_list)} communities")
                try:
                    from scipy.spatial import ConvexHull
                    
                    for i, community in enumerate(community_list):
                        if len(community) > 3:  # Only draw boundary for communities with enough nodes
                            community_nodes = list(community)
                            x_coords = [pos[node][0] for node in community_nodes]
                            y_coords = [pos[node][1] for node in community_nodes]
                            
                            points = list(zip(x_coords, y_coords))
                            if len(points) >= 3:
                                hull = ConvexHull(points)
                                # Get community color for boundary
                                comm_id = sorted(communities.keys())[i]
                                boundary_color = comm_color_map.get(comm_id, 'gray')
                                # Draw hull boundary
                                for simplex in hull.simplices:
                                    ax3.plot([points[simplex[0]][0], points[simplex[1]][0]], 
                                            [points[simplex[0]][1], points[simplex[1]][1]], 
                                            '--', alpha=0.3, color=boundary_color)
                except ImportError:
                    print("    Scipy not available for boundary drawing")
                except Exception as boundary_error:
                    print(f"    Boundary drawing failed: {boundary_error}")
        else:
            ax3.text(0.5, 0.5, f'Graph too large\n({nx_graph.number_of_nodes()} nodes)\nfor visualization', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        boundary_info = " (with boundaries)" if draw_boundaries else " (no boundaries)"
        ax3.set_title(f'Network Graph (Community Layout){boundary_info}\n{nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges')
        ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, 'Empty graph', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Network Graph')
        ax3.axis('off')
    
    # Create legend for communities using chosen colors
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=f'Community {comm_id} ({len(comm_nodes[comm_id])})') 
                      for comm_id, color in comm_color_map.items()]
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.02), 
              ncol=min(len(unique_communities), 6), title=color_info)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SNAPSHOT SUMMARY{title_suffix}")
    print(f"{'='*60}")
    print(f"Communities: {len(unique_communities)}")
    print(f"Nodes: {nx_graph.number_of_nodes()}")
    print(f"Edges: {nx_graph.number_of_edges()}")
    print(f"Modularity: {community_info['modularity']:.3f}")
    print(f"EchoGAE ECI: {data['echogae_eci']:.4f}")
    print(f"DebateGNN ECI: {data['debgnn_eci']:.4f}")
    print(f"Color scheme: {color_info}")
    print(f"Boundaries drawn: {draw_boundaries}")
    
    for comm_id, nodes in comm_nodes.items():
        echo_comm_eci = data['echogae_comm_eci'][list(unique_communities).index(comm_id)]
        debgnn_comm_eci = data['debgnn_comm_eci'][list(unique_communities).index(comm_id)]
        print(f"  Community {comm_id}: {len(nodes)} nodes, EchoGAE ECI: {echo_comm_eci:.4f}, DebateGNN ECI: {debgnn_comm_eci:.4f}")


def analyze_embedding_community_alignment(embeddings, comm_labels, comm_nodes, method_name):
    """Check how well embeddings align with community structure"""
    try:
        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import pdist, squareform
        
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
        
        print(f"{method_name} Embedding-Community Alignment:")
        print(f"  Silhouette Score: {sil_score:.4f} (higher = better separation)")
        print(f"  Avg Intra-Community Distance: {avg_intra:.4f}")
        print(f"  Avg Inter-Community Distance: {avg_inter:.4f}")
        print(f"  Distance Ratio (Inter/Intra): {avg_inter/avg_intra:.4f}" if avg_intra > 0 else "  Distance Ratio: inf")
        
        return {
            'silhouette': sil_score,
            'avg_intra_dist': avg_intra,
            'avg_inter_dist': avg_inter,
            'distance_ratio': avg_inter/avg_intra if avg_intra > 0 else float('inf')
        }
    except ImportError as e:
        print(f"Cannot analyze embedding alignment: {e}")
        return None

def _get_node_colors(processed_dict_single_subreddit, evolution_data=None, color_mode="lineage"):
    """Helper function to get node colors for all timesteps consistently
    
    Args:
        processed_dict_single_subreddit: Dictionary containing processed data for a single subreddit
        evolution_data: Optional evolution data for lineage coloring
        color_mode: Color mode ("lineage", "unique", "default")
    
    Returns:
        dict: Mapping from "{timestep}_{community_id}" to hex color
    """
    if color_mode == "lineage" and evolution_data is not None and build_lineage_colors is not None:
        print("  Using lineage colors for network evolution grid")
        node_color_map = build_lineage_colors(
            processed_dict_single_subreddit,
            evolution_data,
            mode="hungarian",
            min_jaccard_parent=0.05,
            split_strategy="new_hues"
        )
        return node_color_map
        
    elif color_mode == "unique" and build_unique_node_colors is not None:
        print("  Using unique colors for network evolution grid")
        node_color_map = build_unique_node_colors(processed_dict_single_subreddit)
        return node_color_map
        
    else:  # color_mode == "default" or fallback
        print("  Using default color cycling for network evolution grid")
        # Simple color cycling using matplotlib default colors
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        node_color_map = {}
        color_idx = 0
        
        # Get all unique communities across all timesteps
        all_communities = set()
        for ts_data in processed_dict_single_subreddit.values():
            all_communities.update(ts_data['community_info']['comm_nodes'].keys())
        
        # Assign colors to communities
        for ts, ts_data in processed_dict_single_subreddit.items():
            comm_nodes = ts_data['community_info']['comm_nodes']
            for comm_id in comm_nodes.keys():
                color_key = f"{ts}_{comm_id}"
                if color_key not in node_color_map:
                    node_color_map[color_key] = default_colors[color_idx % len(default_colors)]
                    color_idx += 1
        
        return node_color_map

def plot_network_evolution_grid(processed_dict_single_subreddit, evolution_data=None, 
                               color_mode="lineage", figsize_per_plot=(4, 3), save_path=None):
    """Generate a subplot grid showing network graphs for all timesteps
    
    Args:
        processed_dict_single_subreddit: Dictionary containing processed data for a single subreddit
        evolution_data: Optional evolution data for coloring
        color_mode: Color mode for community visualization ("lineage", "unique", "default")
        figsize_per_plot: Size of each individual subplot (width, height)
        save_path: Optional path to save the figure
    
    Returns:
        fig: The matplotlib figure object
    """
    timesteps = sorted(processed_dict_single_subreddit.keys())
    if not timesteps:
        print("No timesteps found in data")
        return None
    
    subreddit_name = processed_dict_single_subreddit[timesteps[0]]['community_info']['subreddit']
    n_timesteps = len(timesteps)
    
    # Calculate grid dimensions
    n_cols = min(4, n_timesteps)  # Maximum 4 columns
    n_rows = (n_timesteps + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    total_width = figsize_per_plot[0] * n_cols
    total_height = figsize_per_plot[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height))
    
    # Handle single subplot case
    if n_timesteps == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'Network Evolution Over Time - r/{subreddit_name}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Get colors for all timesteps using the helper function
    node_colors = _get_node_colors(processed_dict_single_subreddit, evolution_data, color_mode)
    
    print(f"Generating network evolution grid with {len(timesteps)} timesteps")
    print(f"Using color mode: {color_mode}")
    
    for i, ts in enumerate(timesteps):
        ax = axes[i]
        ts_data = processed_dict_single_subreddit[ts]
        
        # Get graph data
        nx_graph = ts_data.get('nx_graph')
        if nx_graph is None or nx_graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, f'No graph data\nfor T{ts}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'T{ts}', fontsize=10, fontweight='bold')
            ax.axis('off')
            continue
        
        # Get community info
        comm_nodes = ts_data['community_info']['comm_nodes']
        partition = ts_data['community_info']['partition']  # user_id -> comm_id mapping
        
        # Create node color mapping for NetworkX (map user_ids to colors)
        nx_node_colors = []
        for node in nx_graph.nodes():
            if node in partition:
                comm_id = partition[node]
                color_key = f"{ts}_{comm_id}"
                color = node_colors.get(color_key, '#808080')
                nx_node_colors.append(color)
            else:
                nx_node_colors.append('#808080')  # Gray for nodes without community assignment
        
        # Get node positions based on graph size
        n_nodes = nx_graph.number_of_nodes()
        if n_nodes < 50:
            pos = nx.spring_layout(nx_graph, k=1.5, iterations=50, seed=42)
            node_size = 30
            edge_alpha = 0.5
            edge_width = 0.8
        elif n_nodes < 200:
            pos = nx.spring_layout(nx_graph, k=1.0, iterations=30, seed=42)
            node_size = 20
            edge_alpha = 0.4
            edge_width = 0.6
        elif n_nodes < 500:
            pos = nx.spring_layout(nx_graph, k=0.5, iterations=20, seed=42)
            node_size = 15
            edge_alpha = 0.3
            edge_width = 0.4
        else:
            # For very large graphs, use a simpler layout
            pos = nx.spring_layout(nx_graph, k=0.3, iterations=10, seed=42)
            node_size = 10
            edge_alpha = 0.2
            edge_width = 0.3
        
        # Draw the network
        nx.draw_networkx_nodes(nx_graph, pos, node_color=nx_node_colors, 
                             node_size=node_size, alpha=0.8, ax=ax, 
                             edgecolors='black', linewidths=0.3)
        nx.draw_networkx_edges(nx_graph, pos, alpha=edge_alpha, width=edge_width, ax=ax,
                             edge_color='#404040')
        
        # Add statistics text box with both EchoGAE and DebateGNN ECS
        n_edges = nx_graph.number_of_edges()
        n_comms = len(comm_nodes)
        
        # Find largest community
        largest_comm_id = max(comm_nodes.keys(), key=lambda x: len(comm_nodes[x]))
        largest_comm_size = len(comm_nodes[largest_comm_id])
        
        # Get both ECS scores
        echogae_ecs = ts_data.get('echogae_eci', 0)
        debgnn_ecs = ts_data.get('debgnn_eci', 0)
        
        # Create stats text with both ECS values and largest community info
        stats_text = (f'Nodes = {n_nodes} | Edges = {n_edges}\n'
                     f'Comms = {n_comms} | Largest: C{largest_comm_id} ({largest_comm_size})\n'
                     f'ECS(echo) = {echogae_ecs:.3f}\n'
                     f'ECS(deb) = {debgnn_ecs:.3f}')
        
        # Position the text box outside the plot area to the right
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                        edgecolor='gray', linewidth=0.5))
        
        ax.set_title(f'T{ts}', fontsize=11, fontweight='bold', pad=5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Hide extra subplots if any
    for i in range(n_timesteps, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network evolution grid saved to: {save_path}")
    
    plt.show()
    return fig