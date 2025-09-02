import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Optional

def plot_user_migration_stats(evolution_data, subreddit_name=None, figsize=(12, 5), save_figures=False, output_dir='results'):
    """Create bar/line plots to visualize user migration statistics over time for a single subreddit."""
    import matplotlib.pyplot as plt
    
    if evolution_data is None:
        print("No evolution data provided")
        return
    
    # Debug: Check the type and content of evolution_data
    print(f"Debug - evolution_data type: {type(evolution_data)}")
    if isinstance(evolution_data, dict):
        print(f"Debug - evolution_data keys: {list(evolution_data.keys())}")
    else:
        print(f"Debug - evolution_data content: {evolution_data}")
        return
    
    # Extract data
    ts_pairs = evolution_data['ts_pairs']
    migration_stats = evolution_data['migration_stats']
    
    if not ts_pairs:
        print("No timestep pairs found")
        return
    
    # Prepare data for plotting
    transitions = [f"T{t1}→T{t2}" for t1, t2 in ts_pairs]
    
    # Extract migration statistics
    retained = [stats['retained'] for stats in migration_stats]
    new_users = [stats['new'] for stats in migration_stats]
    lost_users = [stats['lost'] for stats in migration_stats]
    retention_rates = [stats['retention_rate'] * 100 for stats in migration_stats]
    growth_rates = [stats['growth_rate'] * 100 for stats in migration_stats]
    
    # Create subplots (1x2 layout)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'User Migration Statistics Over Time{f" - r/{subreddit_name}" if subreddit_name else "?"}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Absolute User Counts (Stacked Bar)
    x = np.arange(len(transitions))
    
    ax1.bar(x, retained, label='Retained', alpha=0.8, color='green')
    ax1.bar(x, new_users, bottom=retained, label='New Users', alpha=0.8, color='blue')
    ax1.bar(x, lost_users, bottom=[r + n for r, n in zip(retained, new_users)], 
            label='Lost Users', alpha=0.8, color='red')

    ax1.set_xlabel('Time Transitions', fontsize=9, labelpad=10)
    ax1.set_ylabel('Number of Users', fontsize=9)
    ax1.set_title('(A) User Counts by Category', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(transitions, rotation=45, fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Retention and Growth Rates (Line Plot)
    ax2.plot(x, retention_rates, 'o-', linewidth=2, markersize=6, label='Retention Rate', color='green')
    ax2.plot(x, growth_rates, 's-', linewidth=2, markersize=6, label='Growth Rate', color='blue')
    
    ax2.set_xlabel('Time Transitions', fontsize=9, labelpad=10)
    ax2.set_ylabel('Rate (%)', fontsize=9)
    ax2.set_title('(B) Retention and Growth Rates', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(transitions, rotation=45, fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_figures:
        plt.savefig(f'{output_dir}/user_migration_stats.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig


def plot_combined_migration_and_ecs(evolution_data, processed_dict_single_subreddit, subreddit_name=None, figsize=(20, 10)):
    """Create comprehensive plots combining user migration stats with ECS comparisons for EchoGAE and DebateGNN."""
    if evolution_data is None or not processed_dict_single_subreddit:
        print("No evolution data or processed data provided")
        return
    
    # Extract migration data
    ts_pairs = evolution_data['ts_pairs']
    migration_stats = evolution_data['migration_stats']
    transitions = [f"T{t1}→T{t2}" for t1, t2 in ts_pairs]

    # Extract ECS data from processed_dict
    timesteps = sorted(processed_dict_single_subreddit.keys())
    echogae_ecs_values = [processed_dict_single_subreddit[ts]['echogae_eci'] for ts in timesteps]
    debgnn_ecs_values = [processed_dict_single_subreddit[ts]['debgnn_eci'] for ts in timesteps]
    
    # Calculate ECS changes between timesteps (for transitions)
    echogae_ecs_deltas = [echogae_ecs_values[i+1] - echogae_ecs_values[i] for i in range(len(echogae_ecs_values)-1)]
    debgnn_ecs_deltas = [debgnn_ecs_values[i+1] - debgnn_ecs_values[i] for i in range(len(debgnn_ecs_values)-1)]
    
    # Extract migration statistics
    retention_rates = [stats['retention_rate'] * 100 for stats in migration_stats]
    growth_rates = [stats['growth_rate'] * 100 for stats in migration_stats]
    total_jaccards = evolution_data['total_jaccards']
    
    # Create subplots (2x3 layout)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Community Evolution & ECI Analysis{f" - {subreddit_name}" if subreddit_name else ""}', fontsize=16, fontweight='bold')
    
    x = np.arange(len(transitions))
    
    # Plot 1: ECS Values Over Time (Line Plot)
    timestep_labels = [f"T{ts}" for ts in timesteps]
    ax1.plot(timesteps, echogae_ecs_values, linewidth=2.5, markersize=8, 
             label='EchoGAE ECS', color='blue', alpha=0.8, marker='o')
    ax1.plot(timesteps, debgnn_ecs_values, linewidth=2.5, markersize=8, 
             label='DebateGNN ECS', color='green', alpha=0.8, marker='s')
    
    # Add secondary y-axis for Total Jaccard
    ax1_jaccard = ax1.twinx()
    
    # For Jaccard, we need to map it to transitions (which are between timesteps)
    # We'll plot it at the midpoint between consecutive timesteps
    jaccard_x_positions = [(timesteps[i] + timesteps[i+1])/2 for i in range(len(timesteps)-1)]
    ax1_jaccard.plot(jaccard_x_positions, total_jaccards, linewidth=2.5, markersize=8,
                     label='Total Jaccard', color='purple', alpha=0.8, marker='^', linestyle='--')
    
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Echo Chamber Index', color='black')
    ax1_jaccard.set_ylabel('Total Jaccard Similarity', color='purple')
    ax1.set_title('ECS Values Over Time with Community Stability')
    ax1.set_xticks(timesteps)
    ax1.set_xticklabels(timestep_labels, rotation=45)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_jaccard.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.grid(True, alpha=0.3)

    # Plot 2: ECS Changes vs Jaccard Similarity (Scatter + Line)
    ax2_line = ax2
    ax2_scatter = ax2.twinx()
    
    # Line plot for total Jaccard similarity
    line = ax2_line.plot(x, total_jaccards, linewidth=2, marker='o', markersize=6, 
                         label='Total Jaccard', alpha=0.7, color='purple')
    
    # Scatter plot for ECS changes
    scatter1 = ax2_scatter.scatter(x, echogae_ecs_deltas, c='blue', s=80, alpha=0.7, 
                                   marker='o', label='EchoGAE Δ')
    scatter2 = ax2_scatter.scatter(x, debgnn_ecs_deltas, c='green', s=80, alpha=0.7, 
                                   marker='s', label='DebateGNN Δ')
    
    ax2_line.set_xlabel('Time Transitions')
    ax2_line.set_ylabel('Total Jaccard Similarity', color='green')
    ax2_scatter.set_ylabel('ECS Change (Δ)', color='black')
    ax2_line.set_title('Community Stability vs ECS Changes')
    ax2_line.set_xticks(x)
    ax2_line.set_xticklabels(transitions, rotation=45)
    ax2_scatter.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Combine legends
    lines1, labels1 = ax2_line.get_legend_handles_labels()
    lines2, labels2 = ax2_scatter.get_legend_handles_labels()
    ax2_line.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2_line.grid(True, alpha=0.3)
    
    # Plot 3: User Retention vs ECS (Comparison)
    ax3.plot(x, retention_rates, color="orange", linewidth=2, markersize=6, label='Retention Rate', alpha=0.8)

    # Secondary y-axis for average ECS during transition
    ax3_ecs = ax3.twinx()
    avg_echogae_ecs = [(echogae_ecs_values[i] + echogae_ecs_values[i+1])/2 for i in range(len(echogae_ecs_values)-1)]
    avg_debgnn_ecs = [(debgnn_ecs_values[i] + debgnn_ecs_values[i+1])/2 for i in range(len(debgnn_ecs_values)-1)]
    
    ax3_ecs.plot(x, avg_echogae_ecs, 'blue', linestyle='--', linewidth=2, 
                 marker='o', markersize=4, label='Avg EchoGAE ECS', alpha=0.7)
    ax3_ecs.plot(x, avg_debgnn_ecs, 'green', linestyle='--', linewidth=2,
                 marker='s', markersize=4, label='Avg DebateGNN ECS', alpha=0.7)

    ax3.set_xlabel('Time Transitions')
    ax3.set_ylabel('Retention Rate (%)', color='green')
    ax3_ecs.set_ylabel('Average ECS', color='black')
    ax3.set_title('User Retention vs ECS Levels')
    ax3.set_xticks(x)
    ax3.set_xticklabels(transitions, rotation=45)
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_ecs.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: ECS Delta Comparison (Bar Chart)
    width = 0.35
    ax4.bar(x - width/2, echogae_ecs_deltas, width, label='EchoGAE ΔECI', 
            alpha=0.8, color='blue')
    ax4.bar(x + width/2, debgnn_ecs_deltas, width, label='DebateGNN ΔECI', 
            alpha=0.8, color='green')

    ax4.set_xlabel('Time Transitions')
    ax4.set_ylabel('ECS Change (Δ)')
    ax4.set_title('ECS Changes by Method')
    ax4.set_xticks(x)
    ax4.set_xticklabels(transitions, rotation=45)
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Growth Rate vs ECI Changes (Scatter)
    ax5.scatter(echogae_ecs_deltas, growth_rates, c='blue', s=100, alpha=0.7, 
                marker='o', label='EchoGAE', edgecolors='black', linewidth=0.5)
    ax5.scatter(debgnn_ecs_deltas, growth_rates, c='green', s=100, alpha=0.7, 
                marker='s', label='DebateGNN', edgecolors='black', linewidth=0.5)
    
    # Add transition labels to points
    for i, transition in enumerate(transitions):
        ax5.annotate(transition, (echogae_ecs_deltas[i], growth_rates[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

    ax5.set_xlabel('ECS Change (Δ)')
    ax5.set_ylabel('Growth Rate (%)')
    ax5.set_title('Community Growth vs ECS Changes')
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Method Comparison Summary (Heatmap-style)
    # Create correlation matrix between metrics
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
    
    # Prepare data for correlation analysis
    metrics_df = pd.DataFrame({
        'Retention_Rate': retention_rates,
        'Growth_Rate': growth_rates,
        'Total_Jaccard': total_jaccards,
        'EchoGAE_Delta': echogae_ecs_deltas,
        'DebateGNN_Delta': debgnn_ecs_deltas
    })
    
    correlation_matrix = metrics_df.corr()
    
    # Create heatmap
    im = ax6.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

    # Set ticks and labels
    ax6.set_xticks(range(len(correlation_matrix.columns)))
    ax6.set_yticks(range(len(correlation_matrix.columns)))
    ax6.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax6.set_yticklabels(correlation_matrix.columns)
    ax6.set_title('Metric Correlations')
    
    # Add correlation values as text
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = ax6.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax6, shrink=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive summary
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE ANALYSIS SUMMARY{f' - {subreddit_name}' if subreddit_name else ''}")
    print(f"{'='*70}")
    
    print(f"\nECS EVOLUTION:")
    for i, ts in enumerate(timesteps):
        print(f"  + T{ts}: EchoGAE={echogae_ecs_values[i]:.4f}, DebateGNN={debgnn_ecs_values[i]:.4f}")

    print(f"\nTRANSITION ANALYSIS:")
    for i, (t1, t2) in enumerate(ts_pairs):
        print(f"\nTransition T{t1} → T{t2}:")
        print(f"  + User Retention: {retention_rates[i]:.1f}%")
        print(f"  + User Growth: {growth_rates[i]:.1f}%")
        print(f"  + Total Jaccard: {total_jaccards[i]:.3f}")
        print(f"  + EchoGAE ΔECS: {echogae_ecs_deltas[i]:+.4f}")
        print(f"  + DebateGNN ΔECS: {debgnn_ecs_deltas[i]:+.4f}")

    print(f"\nCORRELATION INSIGHTS:")
    print(f"  + EchoGAE ΔECS vs Retention: {correlation_matrix.loc['EchoGAE_Delta', 'Retention_Rate']:.3f}")
    print(f"  + DebateGNN ΔECS vs Retention: {correlation_matrix.loc['DebateGNN_Delta', 'Retention_Rate']:.3f}")
    print(f"  + EchoGAE ΔECS vs Jaccard: {correlation_matrix.loc['EchoGAE_Delta', 'Total_Jaccard']:.3f}")
    print(f"  + DebateGNN ΔECS vs Jaccard: {correlation_matrix.loc['DebateGNN_Delta', 'Total_Jaccard']:.3f}")

