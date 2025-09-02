import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats

from src.analysis.community_evolution_analyzer import CommunityEvolutionAnalyzer
from src.visualization.ecs_plots import ECSPlotter


class ECSAnalyzer:
    """
    Main class for complete ECS analysis pipeline including:
    - Analysis of pre-computed ECS results
    - Community evolution analysis
    - Statistical comparisons
    - Results figure generation
    """
    
    def __init__(self, output_dir: str = 'results/ecs'):
        self.output_dir = output_dir        
        self.evolution_analyzer = CommunityEvolutionAnalyzer(
            verbose=True,
            matrix_display_limit=10
        )
        self.plotter = ECSPlotter(output_dir=output_dir)
        
        # Storage for results
        self.processed_dict = {}
        self.ecs_df = None
        self.evolution_data = {}
    
    def run_complete_analysis(
        self, 
        processed_dict: Dict,
        ecs_df: pd.DataFrame
    ) -> Dict:
        """
        Run the complete ECS analysis pipeline with pre-computed results
        
        Args:
            processed_dict: Dictionary of processed ECS results
            ecs_df: DataFrame of ECS results
            
        Returns:
            results_dict: Complete analysis results
        """
        print("Starting complete ECS analysis with pre-computed results...")
        
        # Store the pre-computed results
        self.processed_dict = processed_dict
        self.ecs_df = ecs_df
        
        # Analyze community evolution for each subreddit
        print("Analyzing community evolution...")
        for subreddit_id, subreddit_data in self.processed_dict.items():
            evolution_data = self.evolution_analyzer.analyze_evolution(subreddit_data)
            if evolution_data:
                self.evolution_data[subreddit_id] = evolution_data
        
        # Generate comprehensive results
        results = self._compile_results()
        
        return results
    
    def _compile_results(self) -> Dict:
        """Compile all analysis results into a single dictionary"""
        return {
            'processed_dict': self.processed_dict,
            'ecs_dataframe': self.ecs_df,
            'evolution_data': self.evolution_data,
            'statistics': self._compute_statistics(),
            'best_subreddit_id': self._find_best_subreddit()
        }
    
    def _compute_statistics(self) -> Dict:
        """Compute key statistics for results"""
        if self.ecs_df is None:
            return {}
        
        echogae_values = np.array(self.ecs_df['echogae_eci'])
        debgnn_values = np.array(self.ecs_df['debgnn_eci'])
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(echogae_values, debgnn_values)
        diff = echogae_values - debgnn_values
        cohens_d = diff.mean() / diff.std()
        
        return {
            'echogae_mean': echogae_values.mean(),
            'echogae_std': echogae_values.std(),
            'debgnn_mean': debgnn_values.mean(),
            'debgnn_std': debgnn_values.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
    
    def _find_best_subreddit(self) -> int:
        """Find subreddit with most timesteps for detailed analysis"""
        return max(self.processed_dict.keys(), 
                  key=lambda x: len(self.processed_dict[x]))
    
    def generate_result_figures(self, results: Dict, target_subreddit_id: Optional[int] = None, 
                      save_figures: bool = True, embedding_timestep: Optional[int] = None, 
                      plot_all_embedding_timesteps: bool = False, color_mode: str = "lineage"):
        """
        Generate all figures needed for the results
        """
        print("Generating result figures...")
        print(f"Using color mode: {color_mode}")
        
        # Use specified subreddit or fall back to best one
        if target_subreddit_id is not None and target_subreddit_id in results['processed_dict']:
            analysis_sub_id = target_subreddit_id
            print(f"Using specified subreddit ID: {analysis_sub_id}")
        else:
            analysis_sub_id = results['best_subreddit_id']
            if target_subreddit_id is not None:
                print(f"Warning: Subreddit ID {target_subreddit_id} not found, using best: {analysis_sub_id}")
            else:
                print(f"Using best subreddit ID: {analysis_sub_id}")
        
        # Figure 1: Method Comparison - Updated to pass evolution data and target subreddit ID
        self.plotter.plot_method_comparison(
            results['ecs_dataframe'], 
            results['processed_dict'],
            evolution_data=results['evolution_data'],
            target_subreddit_id=analysis_sub_id,
            save=save_figures
        )
        
        # # Figure 2: Community Evolution (specified subreddit)
        # if analysis_sub_id in results['evolution_data']:
        #     evolution_data = results['evolution_data'][analysis_sub_id]
        #     if isinstance(evolution_data, dict):
        #         self.plotter.plot_community_evolution(
        #             evolution_data,
        #             results['processed_dict'][analysis_sub_id],
        #             save=save_figures
        #         )
        #     else:
        #         print(f"Warning: Invalid evolution_data type {type(evolution_data)}, skipping community evolution plot")
        
        # Figure 5: User Migration Statistics
        if analysis_sub_id in results['evolution_data']:
            evolution_data = results['evolution_data'][analysis_sub_id]
            if isinstance(evolution_data, dict):
                self.plotter.plot_user_migration_stats(
                    evolution_data,
                    results['processed_dict'][analysis_sub_id],
                    save=save_figures
                )
            else:
                print(f"Warning: Invalid evolution_data type {type(evolution_data)}, skipping migration stats plot")

        # Figure 3: Embedding Comparison - Individual plots for each timestep or single plot
        if plot_all_embedding_timesteps:
            self._plot_all_timesteps_individually(
                results['processed_dict'][analysis_sub_id],
                results['evolution_data'].get(analysis_sub_id),
                color_mode,
                save_figures
            )
        else:
            # Plot single timestep
            self.plotter.plot_embedding_comparison(
                results['processed_dict'][analysis_sub_id],
                results['evolution_data'].get(analysis_sub_id),
                timestep=embedding_timestep,
                plot_all_timesteps=False,
                color_mode=color_mode,
                save=save_figures
            )
        
        
        # Figure 4: Community Flow Sankey
        # if analysis_sub_id in results['evolution_data']:
        #     evolution_data = results['evolution_data'][analysis_sub_id]
        #     print(f"Debug: evolution_data type = {type(evolution_data)}")
        #     print(f"Debug: Passing color_mode '{color_mode}' to Sankey plot")
            
        #     if isinstance(evolution_data, dict):
        #         self.plotter.plot_community_flow(
        #             evolution_data,
        #             results['processed_dict'][analysis_sub_id],
        #             save=save_figures,
        #             color_mode=color_mode
        #         )
        #     else:
        #         print(f"Warning: Invalid evolution_data type {type(evolution_data)}, skipping community flow plot")
        
        # Figure 6: Network Evolution Grid - Now using the plotter method
        self.plotter.plot_network_evolution_grid(
            processed_dict_single=results['processed_dict'][analysis_sub_id],
            evolution_data=results['evolution_data'].get(analysis_sub_id),
            color_mode=color_mode,
            save=save_figures,
            figsize_per_plot=(3.5, 2.5)
        )

    def _plot_all_timesteps_individually(self, processed_dict_single: Dict, evolution_data: Optional[Dict], 
                                       color_mode: str, save_figures: bool):
        """Generate individual embedding plots for each timestep"""
        from src.visualization.snapshot_plots import plot_snapshot_analysis
        
        timesteps = sorted(processed_dict_single.keys())
        subreddit_name = processed_dict_single[timesteps[0]]['community_info']['subreddit']
        
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
            
            if save_figures:
                import matplotlib.pyplot as plt
                plt.savefig(f'{self.output_dir}/figure3_embedding_comparison_t{ts}.png', 
                           dpi=300, bbox_inches='tight')
                print(f"    Saved: figure3_embedding_comparison_t{ts}.png")
                plt.close()

    def create_result_summary(self, results: Dict) -> pd.DataFrame:
        """Create summary table for results"""
        # First aggregate to get both mean and std
        summary_stats = results['ecs_dataframe'].groupby('subreddit').agg({
            'timestep': 'count',
            'modularity': ['mean', 'std'],
            'echogae_eci': ['mean', 'std'],
            'debgnn_eci': ['mean', 'std'],
            'delta_echogae_eci': ['mean', 'std'],
            'delta_debgnn_eci': ['mean', 'std'],
            'echogae_silhouette': ['mean', 'std'],
            'debgnn_silhouette': ['mean', 'std']
        })
        
        # Calculate overall statistics (across all subreddits) - avoid MultiIndex
        overall_timesteps = len(results['ecs_dataframe'])
        overall_means = results['ecs_dataframe'][['echogae_eci', 'modularity', 'debgnn_eci', 'delta_echogae_eci', 
                                                'delta_debgnn_eci','echogae_silhouette', 
                                                'debgnn_silhouette']].mean()
        overall_stds = results['ecs_dataframe'][['echogae_eci', 'modularity', 'debgnn_eci', 'delta_echogae_eci', 
                                               'delta_debgnn_eci', 'echogae_silhouette', 
                                               'debgnn_silhouette']].std()
        
        # Create new DataFrame with mean ± std format
        summary_table = pd.DataFrame(index=summary_stats.index)
        
        # Add timestep count (no std needed)
        summary_table['n_timesteps'] = summary_stats[('timestep', 'count')]
        
        # Add mean ± std columns
        columns_to_format = [
             'modularity', 'echogae_eci', 'debgnn_eci', 'delta_echogae_eci', 'delta_debgnn_eci', 'echogae_silhouette', 'debgnn_silhouette'
        ]
        
        for col in columns_to_format:
            mean_vals = summary_stats[(col, 'mean')]
            std_vals = summary_stats[(col, 'std')]
            # Format as "mean ± std" with 3 decimal places
            summary_table[col] = mean_vals.round(3).astype(str) + ' ± ' + std_vals.round(3).astype(str)
        
        # Add overall row
        overall_row = pd.DataFrame(index=['ALL'])
        overall_row['n_timesteps'] = overall_timesteps
        
        for col in columns_to_format:
            mean_val = overall_means[col]
            std_val = overall_stds[col]
            overall_row[col] = f"{mean_val:.3f} ± {std_val:.3f}"
        
        # Concatenate the overall row to the summary table
        summary_table = pd.concat([summary_table, overall_row])
        
        return summary_table
    
    def print_result_summary(self, results: Dict, target_subreddit_id: Optional[int] = None):
        """Print formatted results for summary"""
        stats = results['statistics']
        
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")
        
        print(f"DATASET OVERVIEW:")
        print(f"   + Subreddits analyzed: {len(results['processed_dict'])}")
        print(f"   + Total timesteps: {sum(len(data) for data in results['processed_dict'].values())}")
        print(f"   + Total communities: {results['ecs_dataframe']['num_communities'].sum()}")
        
        print(f"\nMETHOD COMPARISON:")
        print(f"   + EchoGAE ECI: {stats['echogae_mean']:.3f} ± {stats['echogae_std']:.3f}")
        print(f"   + DebateGNN ECI: {stats['debgnn_mean']:.3f} ± {stats['debgnn_std']:.3f}")
        print(f"   + Paired t-test: t = {stats['t_statistic']:.3f}, p = {stats['p_value']:.3f}")
        print(f"   + Effect size (Cohen's d): {stats['cohens_d']:.3f} ({stats['effect_size']})")
        print(f"   + Significant difference: {'Yes' if stats['significant'] else 'No'}")
        
        # Use specified subreddit or fall back to best one
        if target_subreddit_id is not None and target_subreddit_id in results['processed_dict']:
            analysis_sub_id = target_subreddit_id
        else:
            analysis_sub_id = results['best_subreddit_id']
            if target_subreddit_id is not None:
                print(f"\nWarning: Subreddit ID {target_subreddit_id} not found, using best: {analysis_sub_id}")
        
        # Community evolution insights for specified subreddit
        if analysis_sub_id in results['evolution_data']:
            evolution_data = results['evolution_data'][analysis_sub_id]
            total_jaccard_mean = np.mean(evolution_data['total_jaccards'])
            migration_stats = evolution_data['migration_stats']
            avg_retention = np.mean([s['retention_rate'] for s in migration_stats])
            avg_growth = np.mean([s['growth_rate'] for s in migration_stats])
            
            subreddit_name = results['processed_dict'][analysis_sub_id][list(results['processed_dict'][analysis_sub_id].keys())[0]]['community_info']['subreddit']
            
            print(f"\nCOMMUNITY EVOLUTION (r/{subreddit_name} - ID {analysis_sub_id}):")
            print(f"   + Average Jaccard similarity: {total_jaccard_mean:.3f}")
            print(f"   + Average retention rate: {avg_retention:.1%}")
            print(f"   + Average growth rate: {avg_growth:.1%}")
        else:
            print(f"\nNo evolution data available for subreddit ID {analysis_sub_id}")

    def run_complete_analysis_with_figures(
        self,
        processed_dict: Dict,
        ecs_df: pd.DataFrame,
        target_subreddit_id: Optional[int] = None,
        save_figures: bool = True,
        embedding_timestep: Optional[int] = None,
        plot_all_embedding_timesteps: bool = False,
        color_mode: str = "lineage"
    ) -> Dict:
        """
        Run complete analysis and generate figures for specified subreddit
        
        Args:
            processed_dict: Dictionary of processed ECS results
            ecs_df: DataFrame of ECS results
            target_subreddit_id: Optional subreddit ID to focus analysis on
            save_figures: Whether to save generated figures
            embedding_timestep: Specific timestep for embedding comparison (None = mid timestep)
            plot_all_embedding_timesteps: Whether to plot all timesteps for embedding comparison
            color_mode: Color mode for embedding visualization ["lineage", "unique", "default"]

        Returns:
            results_dict: Complete analysis results
        """
        print("Running complete ECS analysis with figure generation...")
        print(f"   + Color mode: {color_mode}")

        # Run the standard analysis pipeline
        results = self.run_complete_analysis(processed_dict, ecs_df)
        
        # Print available subreddit options
        print(f"\nAVAILABLE SUBREDDITS:")
        for sub_id, sub_data in results['processed_dict'].items():
            first_ts = list(sub_data.keys())[0]
            sub_name = sub_data[first_ts]['community_info']['subreddit']
            num_timesteps = len(sub_data)
            print(f"   + ID {sub_id}: r/{sub_name} ({num_timesteps} timesteps)")
        
        # Generate figures for specified subreddit
        self.generate_result_figures(
            results, 
            target_subreddit_id, 
            save_figures, 
            embedding_timestep, 
            plot_all_embedding_timesteps,
            color_mode=color_mode
        )
        
        # Print summary for specified subreddit
        self.print_result_summary(results, target_subreddit_id)
    
        # Generate and save summary table
        summary_table = self.create_result_summary(results)
        display(summary_table)
        if save_figures:
            summary_table.to_csv(f'{self.output_dir}/ecs_summary_table.csv')
            print(f"\nSummary table saved to: {self.output_dir}/ecs_summary_table.csv")
        
        return results