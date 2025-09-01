import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional


class ECSPlotter:
    """Handles all ECS-related plotting and visualization"""

    def __init__(self, output_dir: str = 'results/ecs'):
        self.output_dir = output_dir
    
    def plot_method_comparison(self, ecs_df: pd.DataFrame, processed_dict: Dict, save: bool = False):
        """Generate Figure 1: Method Comparison (EchoGAE vs DebateGNN)"""
        print("Generating Figure 1: Method Comparison...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Figure 1A: Overall ECS comparison
        methods = ['EchoGAE', 'DebateGNN']
        echo_mean, echo_std = ecs_df['echogae_eci'].mean(), ecs_df['echogae_eci'].std()
        gnn_mean, gnn_std = ecs_df['debgnn_eci'].mean(), ecs_df['debgnn_eci'].std()
        mean_values = [echo_mean, gnn_mean]
        std_values = [echo_std, gnn_std]
        
        bars = ax1.bar(methods, mean_values, yerr=std_values, capsize=8, alpha=0.8, 
                       color=['blue', 'green'], edgecolor='black', linewidth=1)
        ax1.set_ylabel('Echo Chamber Index', fontsize=12)
        ax1.set_title('(A) ECS Comparison by Method', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(mean_values) * 1.3)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, mean_values, std_values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                     f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Figure 1B: ECS evolution over time by subreddit
        for sub_id in processed_dict.keys():
            timesteps = sorted(processed_dict[sub_id].keys())
            if len(timesteps) > 1:
                subreddit_name = processed_dict[sub_id][timesteps[0]]['community_info']['subreddit']
                echogae_values = [processed_dict[sub_id][ts]['echogae_eci'] for ts in timesteps]
                debgnn_values = [processed_dict[sub_id][ts]['debgnn_eci'] for ts in timesteps]
                
                ax2.plot(timesteps, echogae_values, 'o-', alpha=0.8, linewidth=2,
                        label=f'{subreddit_name} (EchoGAE)', color='blue')
                ax2.plot(timesteps, debgnn_values, 's--', alpha=0.8, linewidth=2,
                        label=f'{subreddit_name} (DebateGNN)', color='green')
        
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Echo Chamber Index', fontsize=12)
        ax2.set_title('(B) ECS Evolution Over Time', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
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