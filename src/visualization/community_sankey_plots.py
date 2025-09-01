import plotly.graph_objects as go
import numpy as np
from typing import Dict, Optional
from src.utils.community_colors import build_lineage_colors, build_unique_node_colors


def create_community_sankey(evolution_data, processed_dict_single_subreddit, subreddit_name=None, color_mode: str = "lineage"):
    """
    Create a Sankey diagram showing Hungarian algorithm optimal community matches with Jaccard values.
    """
    if evolution_data is None or not processed_dict_single_subreddit:
        print("No evolution data or processed data provided")
        return
    
    timesteps = sorted(processed_dict_single_subreddit.keys())
    ts_pairs = evolution_data['ts_pairs']
    hungarian_matches = evolution_data['hungarian_matches']
    
    if color_mode == "unique":
        node_color_map = build_unique_node_colors(processed_dict_single_subreddit)
    else:
        node_color_map = build_lineage_colors(
            processed_dict_single_subreddit,
            evolution_data,
            mode="hungarian",
            min_jaccard_parent=0.05,
            split_strategy="new_hues"
        )

    # Create nodes
    all_nodes, node_colors, node_indices = [], [], {}
    for i, ts in enumerate(timesteps):
        comm_nodes = processed_dict_single_subreddit[ts]['community_info']['comm_nodes']
        for comm_id in sorted(comm_nodes.keys()):
            node_label = f"T{ts}_C{comm_id} ({len(comm_nodes[comm_id])})"
            node_key = f"{ts}_{comm_id}"
            all_nodes.append(node_label)
            node_indices[node_key] = len(all_nodes) - 1
            node_colors.append(node_color_map.get(node_key, "#999999"))
    
    # Create flows from Hungarian matches
    sources = []
    targets = []
    values = []
    jaccard_scores = []
    match_details = []
    
    print("Creating Hungarian Algorithm Sankey diagram...")
    
    for transition_idx, (t1, t2) in enumerate(ts_pairs):
        transition_matches = hungarian_matches[transition_idx]
        print(f"\nTransition T{t1} → T{t2} (Hungarian Optimal Matches):")
        
        matched_flows = [m for m in transition_matches if m['assignment_type'] == 'matched' and m['jaccard'] > 0]
        
        for match in matched_flows:
            t1_comm = match['t1_comm']
            t2_comm = match['t2_comm']
            jaccard = match['jaccard']
            t1_size = match['t1_size']
            t2_size = match['t2_size']
            
            # Calculate actual flow value (intersection of users)
            t1_communities = processed_dict_single_subreddit[t1]['community_info']['comm_nodes']
            t2_communities = processed_dict_single_subreddit[t2]['community_info']['comm_nodes']
            
            t1_users = set(t1_communities[t1_comm])
            t2_users = set(t2_communities[t2_comm])
            flow_value = len(t1_users.intersection(t2_users))
            
            source_key = f"{t1}_{t1_comm}"
            target_key = f"{t2}_{t2_comm}"
            
            if source_key in node_indices and target_key in node_indices:
                sources.append(node_indices[source_key])
                targets.append(node_indices[target_key])
                values.append(flow_value)
                jaccard_scores.append(jaccard)
                
                match_details.append({
                    'transition': f"T{t1}→T{t2}",
                    't1_comm': t1_comm,
                    't2_comm': t2_comm,
                    'flow_value': flow_value,
                    'jaccard': jaccard,
                    't1_size': t1_size,
                    't2_size': t2_size
                })
                
                print(f"    C{t1_comm}→C{t2_comm}: {flow_value} users (J={jaccard:.3f})")
    
    # Create enhanced link colors based on Jaccard similarity
    link_colors = []
    for jaccard in jaccard_scores:
        if jaccard >= 0.5:
            link_colors.append("rgba(0, 128, 0, 0.8)")  # Strong green
        elif jaccard >= 0.3:
            link_colors.append("rgba(255, 165, 0, 0.7)")  # Orange
        elif jaccard >= 0.1:
            link_colors.append("rgba(255, 255, 0, 0.6)")  # Yellow
        else:
            link_colors.append("rgba(128, 128, 128, 0.4)")  # Gray
    
    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}<br>' +
                         'Flow: %{value} users<br>' +
                         'Jaccard: %{customdata:.3f}<br>' +
                         'Assignment: Hungarian Optimal<extra></extra>',
            customdata=jaccard_scores
        )
    ))

    title = f"{subreddit_name}'s Community Flow (Hungarian Best Matches)"
    subtitle = "Link Colors: 🟢 Strong (J≥0.5) 🟠 Medium (0.3≤J<0.5) 🟡 Low (0.1≤J<0.3) ⚫ Very Low (J<0.1)"
    
    annot_node_colors = "Lineage (siblings share hue)" if color_mode != "unique" else "Unique per community"
    fig.update_layout(
        title_text=f"{title}<br><sub>{subtitle}</sub>",
        font_size=12,
        height=700,
        width=1500,  # wider canvas
        margin=dict(l=50, r=50, t=100, b=50),
        annotations=[
            dict(
                text=f"Shows optimal 1-to-1 community assignments maximizing total Jaccard similarity",
                showarrow=False, xref="paper", yref="paper", x=0.5, y=-0.1, xanchor='center', yanchor='top',
                font=dict(size=10, color="gray")
            ),
            dict(
                text=f"Node Colors = {annot_node_colors}",
                showarrow=False, xref="paper", yref="paper", x=0.5, y=-0.16, xanchor='center', yanchor='top',
                font=dict(size=10, color="gray")
            )
        ]
    )
    
    fig.show()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"HUNGARIAN MATCHES SUMMARY{f' - {subreddit_name}' if subreddit_name else ''}")
    print(f"{'='*70}")
    print(f"Total optimal matches: {len(sources)}")
    print(f"Total users in optimal flows: {sum(values)}")
    print(f"Average Jaccard similarity: {np.mean(jaccard_scores):.3f}")
    
    return match_details


def create_all_flows_sankey(evolution_data, processed_dict_single_subreddit, 
                           min_flow_threshold=3, min_jaccard_threshold=0.01, 
                           subreddit_name=None, color_mode: str = "lineage"):
    """Create a Sankey diagram showing ALL significant user flows between communities"""
    if evolution_data is None or not processed_dict_single_subreddit:
        print("No evolution data or processed data provided")
        return
    
    timesteps = sorted(processed_dict_single_subreddit.keys())
    ts_pairs = evolution_data['ts_pairs']
    
    if color_mode == "unique":
        node_color_map = build_unique_node_colors(processed_dict_single_subreddit)
    else:
        node_color_map = build_lineage_colors(
            processed_dict_single_subreddit,
            evolution_data,
            mode="hungarian",
            min_jaccard_parent=0.05,
            split_strategy="new_hues"
        )
    # DEBUG PRINTS
    print("\n=== SANKEY COLOR DEBUG ===")
    print("Sample color mappings from Sankey:")
    for i, (key, color) in enumerate(list(node_color_map.items())[:10]):
        print(f"  {key}: {color}")
    print("=========================\n")

    # Create nodes
    all_nodes, node_colors, node_indices, node_sizes = [], [], {}, []
    for i, ts in enumerate(timesteps):
        comm_nodes = processed_dict_single_subreddit[ts]['community_info']['comm_nodes']
        for comm_id in sorted(comm_nodes.keys()):
            node_label = f"T{ts}_C{comm_id} ({len(comm_nodes[comm_id])})"
            node_key = f"{ts}_{comm_id}"
            all_nodes.append(node_label)
            node_indices[node_key] = len(all_nodes) - 1
            node_sizes.append(len(comm_nodes[comm_id]))
            node_colors.append(node_color_map.get(node_key, "#999999"))
    
    # Create ALL significant flows (not just Hungarian matches)
    sources, targets, values, jaccard_scores, flow_details = [], [], [], [], []
    
    print(f"Analyzing ALL user flows with thresholds:")
    print(f"  + Minimum flow: {min_flow_threshold} users")
    print(f"  + Minimum Jaccard similarity: {min_jaccard_threshold:.3f}")
    
    total_flows_found = 0
    total_flows_filtered = 0
    
    for transition_idx, (t1, t2) in enumerate(ts_pairs):
        print(f"\nTransition T{t1} → T{t2}:")
        
        # Get all communities for both timesteps
        t1_communities = processed_dict_single_subreddit[t1]['community_info']['comm_nodes']
        t2_communities = processed_dict_single_subreddit[t2]['community_info']['comm_nodes']
        
        transition_flows = []
        transition_filtered = 0
        
        # Check ALL community pairs for flows
        for t1_comm, t1_users in t1_communities.items():
            for t2_comm, t2_users in t2_communities.items():
                # Calculate actual user intersection
                t1_user_set = set(t1_users)
                t2_user_set = set(t2_users)
                flow_value = len(t1_user_set.intersection(t2_user_set))
                
                # Calculate Jaccard similarity
                union_size = len(t1_user_set.union(t2_user_set))
                jaccard = flow_value / union_size if union_size > 0 else 0.0
                
                total_flows_found += 1
                
                # Check if flow meets thresholds
                if flow_value >= min_flow_threshold and jaccard >= min_jaccard_threshold:
                    source_key = f"{t1}_{t1_comm}"
                    target_key = f"{t2}_{t2_comm}"
                    
                    if source_key in node_indices and target_key in node_indices:
                        sources.append(node_indices[source_key])
                        targets.append(node_indices[target_key])
                        values.append(flow_value)
                        jaccard_scores.append(jaccard)
                        
                        flow_details.append({
                            'transition': f"T{t1}→T{t2}",
                            't1_comm': t1_comm,
                            't2_comm': t2_comm,
                            'flow_value': flow_value,
                            'jaccard': jaccard,
                            't1_size': len(t1_users),
                            't2_size': len(t2_users)
                        })
                        
                        transition_flows.append((t1_comm, t2_comm, flow_value, jaccard))
                else:
                    transition_filtered += 1
        
        total_flows_filtered += transition_filtered
        
        # Print transition summary
        print(f"  Found {len(transition_flows)} significant flows (filtered out {transition_filtered}):")
        for t1_comm, t2_comm, flow, jacc in sorted(transition_flows, key=lambda x: x[2], reverse=True)[:10]:
            strength = "Strong" if jacc >= 0.5 else "Medium" if jacc >= 0.3 else "Low" if jacc >= 0.1 else "Very Low"
            print(f"    C{t1_comm}→C{t2_comm}: {flow} users (J={jacc:.3f}, {strength})")
        if len(transition_flows) > 10:
            print(f"    ... and {len(transition_flows) - 10} more flows")
    
    # Create enhanced link colors based on Jaccard similarity
    link_colors = []
    significance_labels = []
    for jaccard in jaccard_scores:
        if jaccard >= 0.5:
            link_colors.append("rgba(0, 128, 0, 0.8)")
            significance_labels.append("Strong")
        elif jaccard >= 0.3:
            link_colors.append("rgba(255, 165, 0, 0.7)")
            significance_labels.append("Medium")
        elif jaccard >= 0.1:
            link_colors.append("rgba(255, 255, 0, 0.6)")
            significance_labels.append("Low")
        else:
            link_colors.append("rgba(128, 128, 128, 0.4)")
            significance_labels.append("Very Low")

    # Create the main Sankey diagram
    fig = go.Figure()

    fig.add_trace(go.Sankey(
        name="All Significant Flows",
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=[[j, s] for j, s in zip(jaccard_scores, significance_labels)],
            hovertemplate='%{source.label} → %{target.label}<br>'
                            'Flow: %{value} users<br>'
                            'Jaccard: %{customdata[0]:.3f}<br>'
                            'Significance: %{customdata[1]}<extra></extra>',
        )
    ))
    
    # Create title and subtitle with filtering info
    title = f"{subreddit_name} - Community User Flows" if subreddit_name else ''
    subtitle = (f"Link Colors: 🟢 Strong (J≥0.5) 🟠 Medium (0.3≤J<0.5) 🟡 Low (0.1≤J<0.3) ⚫ Very Low (J<0.1)<br>"
                f"Filters: Min {min_flow_threshold} users, Min J≥{min_jaccard_threshold:.3f} | "
                f"Showing {len(sources)}/{total_flows_found} flows ({total_flows_filtered} filtered out)")
    
    annot_node_colors = "Lineage (siblings share hue)" if color_mode != "unique" else "Unique per community"
    fig.update_layout(
        title_text=f"{title}<br><sub>{subtitle}</sub>",
        font_size=12,
        height=700,
        width=1500,
        margin=dict(l=50, r=50, t=120, b=50),
        annotations=[
            dict(
                text=f"Link Width ∝ User Flow Size | Node Colors = {annot_node_colors} | Shows complete many-to-many community relationships",
                showarrow=False, xref="paper", yref="paper", x=0.5, y=-0.12, xanchor='center', yanchor='top',
                font=dict(size=10, color="gray")
            )
        ]
    )
    
    fig.show()
    
    # Color distribution summary
    color_counts = {'Strong': 0, 'Medium': 0, 'Low': 0, 'Very Low': 0}
    for jaccard in jaccard_scores:
        if jaccard >= 0.5:
            color_counts['Strong'] += 1
        elif jaccard >= 0.3:
            color_counts['Medium'] += 1
        elif jaccard >= 0.1:
            color_counts['Low'] += 1
        else:
            color_counts['Very Low'] += 1
    
    # Print comprehensive summary
    print(f"\n{'='*70}")
    print(f"ALL FLOWS SANKEY SUMMARY{f' - {subreddit_name}' if subreddit_name else ''}")
    print(f"{'='*70}")
    print(f"FILTERING RESULTS:")
    print(f"  + Total possible flows: {total_flows_found:,}")
    print(f"  + Flows after filtering: {len(sources):,}")
    print(f"  + Filtered out: {total_flows_filtered:,} ({total_flows_filtered/total_flows_found*100:.1f}%)")
    print(f"  + Filter efficiency: {len(sources)/total_flows_found*100:.1f}% flows retained")
    
    print(f"\nFLOW STATISTICS:")
    print(f"  + Total nodes (communities): {len(all_nodes)}")
    print(f"  + Total links (significant flows): {len(sources)}")
    print(f"  + Total users in flows: {sum(values):,}")
    print(f"  + Average flow size: {np.mean(values):.1f} users")
    print(f"  + Average Jaccard similarity: {np.mean(jaccard_scores):.3f}")

    print(f"\nFLOW STRENGTH DISTRIBUTION:")
    for strength, count in color_counts.items():
        percentage = count / len(jaccard_scores) * 100 if jaccard_scores else 0
        print(f"  + {strength}: {count} flows ({percentage:.1f}%)")

    # Flow statistics by transition
    for t1, t2 in ts_pairs:
        transition_flows = [f for f in flow_details if f['transition'] == f"T{t1}→T{t2}"]
        total_flow = sum(f['flow_value'] for f in transition_flows)
        avg_jaccard = np.mean([f['jaccard'] for f in transition_flows]) if transition_flows else 0
        print(f"  T{t1} → T{t2}: {len(transition_flows)} flows, {total_flow:,} total users, avg Jaccard: {avg_jaccard:.3f}")
    
    return flow_details


def create_flow_comparison_sankey(evolution_data, processed_dict_single_subreddit, 
                                 min_flow_threshold=3, min_jaccard_threshold=0.01, 
                                 subreddit_name=None):
    """
    Create side-by-side comparison: Hungarian matches vs All significant flows
    """
    print("="*80)
    print("HUNGARIAN ALGORITHM MATCHES (Optimal Assignment)")
    print("="*80)
    
    # Hungarian-based Sankey with Jaccard values
    hungarian_details = create_community_sankey(evolution_data, processed_dict_single_subreddit, 
                                               f"{subreddit_name} - Hungarian Matches" if subreddit_name else "Hungarian Matches")
    
    print("\n" + "="*80)
    print("ALL SIGNIFICANT FLOWS (Complete Picture)")
    print("="*80)
    
    # All flows Sankey with enhanced filtering
    flow_details = create_all_flows_sankey(evolution_data, processed_dict_single_subreddit, 
                                          min_flow_threshold=min_flow_threshold,
                                          min_jaccard_threshold=min_jaccard_threshold,
                                          subreddit_name=f"{subreddit_name} - All Flows" if subreddit_name else "All Flows")
    
    # Comparison analysis
    print("\n" + "="*80)
    print("COMPARISON: Hungarian vs All Flows")
    print("="*80)
    
    if hungarian_details and flow_details:
        hungarian_total = sum(d['flow_value'] for d in hungarian_details)
        all_flows_total = sum(d['flow_value'] for d in flow_details)
        hungarian_avg_jaccard = np.mean([d['jaccard'] for d in hungarian_details])
        all_flows_avg_jaccard = np.mean([d['jaccard'] for d in flow_details])
        
        print(f"COVERAGE COMPARISON:")
        print(f"  + Hungarian flows: {len(hungarian_details)} flows, {hungarian_total:,} users")
        print(f"  + All significant flows: {len(flow_details)} flows, {all_flows_total:,} users")
        print(f"  + Coverage increase: {len(flow_details)/len(hungarian_details):.1f}x flows, {all_flows_total/hungarian_total:.1f}x users")
        
        print(f"\nQUALITY COMPARISON:")
        print(f"  + Hungarian avg Jaccard: {hungarian_avg_jaccard:.3f}")
        print(f"  + All flows avg Jaccard: {all_flows_avg_jaccard:.3f}")
        print(f"  + Quality difference: {hungarian_avg_jaccard - all_flows_avg_jaccard:+.3f}")
    
    return flow_details