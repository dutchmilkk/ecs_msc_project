import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional


class ECSPlotter:
    """Handles all ECS-related plotting and visualization"""

    def __init__(self, output_dir: str = 'results/ecs'):
        self.output_dir = output_dir
    
    def plot_method_comparison(self, ecs_df: pd.DataFrame, processed_dict: Dict, evolution_data: Optional[Dict] = None, 
                             target_subreddit_id: Optional[int] = None, save: bool = False):
        """Generate Figure 1: Method Comparison (EchoGAE vs DebateGNN)"""
        print("Generating Figure 1: Method Comparison...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Determine which subreddit to use for Figure 1B
        if target_subreddit_id is not None and target_subreddit_id in processed_dict:
            analysis_sub_id = target_subreddit_id
        else:
            # Use the subreddit with the most timesteps
            analysis_sub_id = max(processed_dict.keys(), key=lambda x: len(processed_dict[x]))
            if target_subreddit_id is not None:
                print(f"Warning: Subreddit ID {target_subreddit_id} not found, using best: {analysis_sub_id}")
        
        processed_dict_single = processed_dict[analysis_sub_id]
        timesteps_single = sorted(processed_dict_single.keys())
        subreddit_name = processed_dict_single[timesteps_single[0]]['community_info']['subreddit']
        
        # Collect subreddit names for the subtitle
        subreddit_names = []
        legend_added = {"echogae": False, "debgnn": False}
        
        # Figure 1A: ECS evolution over time by subreddit (all subreddits)
        for sub_id in processed_dict.keys():
            timesteps_sub = sorted(processed_dict[sub_id].keys())
            if len(timesteps_sub) > 1:
                subreddit_name_sub = processed_dict[sub_id][timesteps_sub[0]]['community_info']['subreddit']
                subreddit_names.append(f"r/{subreddit_name_sub}")
                echogae_values = [processed_dict[sub_id][ts]['echogae_eci'] for ts in timesteps_sub]
                debgnn_values = [processed_dict[sub_id][ts]['debgnn_eci'] for ts in timesteps_sub]
                
                # Plot EchoGAE line
                echogae_label = 'EchoGAE' if not legend_added["echogae"] else None
                ax1.plot(timesteps_sub, echogae_values, 'o-', alpha=0.8, linewidth=2,
                        label=echogae_label, color='blue')
                legend_added["echogae"] = True
                
                # Plot DebateGNN line
                debgnn_label = 'DebateGNN' if not legend_added["debgnn"] else None
                ax1.plot(timesteps_sub, debgnn_values, 's--', alpha=0.8, linewidth=2,
                        label=debgnn_label, color='green')
                legend_added["debgnn"] = True
        
        # Create subtitle with subreddit names
        subreddits_text = ", ".join(subreddit_names)
        
        ax1.set_xlabel('Timestep', fontsize=12, labelpad=10)
        ax1.set_ylabel('Echo Chamber Index', fontsize=12, labelpad=10)
        ax1.set_title(f'(A) ECS Evolution Over Time (All Subreddits)\n{subreddits_text}', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Figure 1B: ECS Values Over Time with Jaccard (from target subreddit)
        echogae_ecs_values = [processed_dict_single[ts]['echogae_eci'] for ts in timesteps_single]
        debgnn_ecs_values = [processed_dict_single[ts]['debgnn_eci'] for ts in timesteps_single]
        
        timestep_labels = [f"T{ts}" for ts in timesteps_single]
        ax2.plot(timesteps_single, echogae_ecs_values, linewidth=2.5, markersize=8, 
                 label='EchoGAE ECS', color='blue', alpha=0.8, marker='o')
        ax2.plot(timesteps_single, debgnn_ecs_values, linewidth=2.5, markersize=8, 
                 label='DebateGNN ECS', color='green', alpha=0.8, marker='s')
        
        # Add secondary y-axis for Total Jaccard if evolution data is available
        if evolution_data and analysis_sub_id in evolution_data:
            ax2_jaccard = ax2.twinx()
            total_jaccards = evolution_data[analysis_sub_id]['total_jaccards']
            
            # Plot Jaccard at midpoints between timesteps
            jaccard_x_positions = [(timesteps_single[i] + timesteps_single[i+1])/2 for i in range(len(timesteps_single)-1)]
            ax2_jaccard.plot(jaccard_x_positions, total_jaccards, linewidth=2.5, markersize=8,
                             label='Total Jaccard', color='purple', alpha=0.8, marker='^', linestyle='--')
            
            ax2_jaccard.set_ylabel('Total Jaccard Similarity', color='purple', labelpad=10, fontsize=12)
            
            # Combine legends from both axes
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_jaccard.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax2.legend()
        
        ax2.set_xlabel('Timesteps', fontsize=12, labelpad=10)
        # ax2.set_ylabel('Echo Chamber Index', fontsize=12)
        ax2.set_title(f'(B) ECS Evolution Over Time\nr/{subreddit_name}', fontsize=14, fontweight='bold')
        ax2.set_xticks(timesteps_single)
        ax2.set_xticklabels(timestep_labels, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Increase padding between subplots
        plt.subplots_adjust(wspace=0.8)
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/figure1_method_comparison.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_community_evolution(self, evolution_data: Dict, processed_dict_single: Dict, save: bool = False):
        """Generate Figure 2: Community Evolution Analysis"""
        from .community_migration_plots import plot_combined_migration_and_ecs
        
        print("Generating Figure 2: Community Evolution Analysis...")
        subreddit_name = processed_dict_single[list(processed_dict_single.keys())[0]]['community_info']['subreddit']
        plot_combined_migration_and_ecs(
            evolution_data, 
            processed_dict_single, 
            subreddit_name
        )
        
        if save:
            plt.savefig(f'{self.output_dir}/figure2_community_evolution.png', 
                       dpi=300, bbox_inches='tight')
    
    def plot_user_migration_stats(self, evolution_data: Dict, processed_dict_single: Dict, save: bool = False):
        """Generate Figure 5: User Migration Statistics"""
        from .community_migration_plots import plot_user_migration_stats as plot_migration
        
        # Extract subreddit name from processed_dict_single
        first_timestep = list(processed_dict_single.keys())[0]
        subreddit_name = processed_dict_single[first_timestep]['community_info']['subreddit']
        
        # Call the function with the correct signature - only pass evolution_data and subreddit_name
        plot_migration(evolution_data, subreddit_name)
        
        if save:
            plt.savefig(f'{self.output_dir}/figure5_user_migration_stats.png', 
                    dpi=300, bbox_inches='tight')
    
    def plot_embedding_comparison(self, processed_dict_single: Dict, evolution_data: Optional[Dict] = None, 
                                timestep: Optional[int] = None, plot_all_timesteps: bool = False, color_mode: str = "lineage", save: bool = False):
        """Generate Figure 3: Embedding Space Comparison
        
        Args:
            processed_dict_single: Dictionary containing processed data for a single subreddit
            evolution_data: Optional evolution data
            timestep: Specific timestep to plot (if None, uses mid timestep)
            plot_all_timesteps: If True, plots each timestep individually (not subplots)
            color_mode: Color mode for community visualization
            save: Whether to save the figure
        """
        from .snapshot_plots import plot_snapshot_analysis
        
        timesteps = sorted(processed_dict_single.keys())
        subreddit_name = processed_dict_single[timesteps[0]]['community_info']['subreddit']
        
        if plot_all_timesteps:
            # Plot each timestep individually (not as subplots)
            print(f"Plotting embedding comparison for all {len(timesteps)} timesteps individually...")
            
            for ts in timesteps:
                print(f"  - Plotting timestep {ts}")
                plot_snapshot_analysis(
                    processed_dict_single_subreddit=processed_dict_single,
                    timestep=ts,
                    evolution_data=evolution_data,
                    subreddit_name=subreddit_name,
                    color_mode=color_mode,
                    draw_boundaries=False
                )
                
                if save:
                    plt.savefig(f'{self.output_dir}/figure3_embedding_comparison_t{ts}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()  # Close the figure to free memory
                    
        else:
            # Plot single timestep
            if timestep is None:
                # Use mid timestep as default
                target_timestep = timesteps[len(timesteps)//2]
            else:
                # Validate and use provided timestep
                if timestep not in timesteps:
                    print(f"Warning: Timestep {timestep} not found in available timesteps {timesteps}")
                    print(f"Using closest available timestep...")
                    target_timestep = min(timesteps, key=lambda x: abs(x - timestep))
                else:
                    target_timestep = timestep
            
            print(f"Plotting embedding comparison for timestep {target_timestep}")
            
            plot_snapshot_analysis(
                processed_dict_single_subreddit=processed_dict_single,
                timestep=target_timestep,
                evolution_data=evolution_data,
                subreddit_name=subreddit_name,
                color_mode=color_mode,
                draw_boundaries=False
            )
            
            if save:
                plt.savefig(f'{self.output_dir}/figure3_embedding_comparison_t{target_timestep}.png', 
                           dpi=300, bbox_inches='tight')

    def plot_community_flow(self, evolution_data: Dict, processed_dict_single: Dict, color_mode: str = "lineage", save: bool = False):
        """Generate Figure 4: Community Flow Sankey Diagram"""
        from .community_sankey_plots import create_all_flows_sankey
        
        subreddit_name = processed_dict_single[list(processed_dict_single.keys())[0]]['community_info']['subreddit']
        
        create_all_flows_sankey(
            evolution_data=evolution_data,
            processed_dict_single_subreddit=processed_dict_single,
            min_flow_threshold=3,
            min_jaccard_threshold=0.05,
            subreddit_name=f"r/{subreddit_name} - Community Flow Analysis",
            color_mode=color_mode
        )
        
        # Note: Plotly figures need different saving approach
        if save:
            print(f"Sankey diagram displayed (save manually from browser for paper)")
    
    def plot_network_evolution_grid(self, processed_dict_single: Dict, evolution_data: Optional[Dict] = None, 
                                   color_mode: str = "lineage", save: bool = False, figsize_per_plot: tuple = (4, 3)):
        """Generate Figure 6: Network Evolution Grid"""
        from .snapshot_plots import plot_network_evolution_grid as plot_grid_impl
        
        print("Generating Figure 6: Network Evolution Grid...")
        save_path = f'{self.output_dir}/figure6_network_evolution_grid.png' if save else None
        
        return plot_grid_impl(
            processed_dict_single_subreddit=processed_dict_single,
            evolution_data=evolution_data,
            color_mode=color_mode,
            figsize_per_plot=figsize_per_plot,
            save_path=save_path
        )