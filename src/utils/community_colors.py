import colorsys
from typing import Dict, List, Optional
import random

def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def _lighten_color(hex_color: str, amount: float = 0.15) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    r_f, g_f, b_f = [c/255.0 for c in (r, g, b)]
    h, l, s = colorsys.rgb_to_hls(r_f, g_f, b_f)
    l = min(1.0, l + amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex((int(round(r2*255)), int(round(g2*255)), int(round(b2*255))))

def _shift_hue(hex_color: str, degrees: float) -> str:
    # shift hue by degrees in HLS space
    r, g, b = _hex_to_rgb(hex_color)
    r_f, g_f, b_f = [c/255.0 for c in (r, g, b)]
    h, l, s = colorsys.rgb_to_hls(r_f, g_f, b_f)
    h = (h + degrees/360.0) % 1.0
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex((int(round(r2*255)), int(round(g2*255)), int(round(b2*255))))


def _generate_distinct_colors(n: int, saturation: float = 0.7, lightness: float = 0.6) -> List[str]:
    """Generate n visually distinct colors using HSL color space"""
    colors = []
    golden_angle = 137.508  # Golden angle in degrees for maximum color separation
    
    for i in range(n):
        # Use golden angle to maximize hue separation
        hue = (i * golden_angle) % 360
        h_norm = hue / 360.0
        
        # Convert HSL to RGB
        r, g, b = colorsys.hls_to_rgb(h_norm, lightness, saturation)
        
        # Convert to hex
        hex_color = '#{:02X}{:02X}{:02X}'.format(
            int(round(r * 255)), 
            int(round(g * 255)), 
            int(round(b * 255))
        )
        colors.append(hex_color)
    
    return colors

def _get_palette(name: str = "tableau20"):
    if name == "glasbey32":
        return [
            "#0000FF", "#FF0000", "#00FF00", "#000080", "#FF00FF", "#008000", "#800000", "#00FFFF",
            "#808080", "#800080", "#FFFF00", "#008080", "#000000", "#FFA500", "#A52A2A", "#DEB887",
            "#5F9EA0", "#7FFF00", "#D2691E", "#FF7F50", "#6495ED", "#FFF8DC", "#DC143C", "#00FFFF",
            "#00008B", "#008B8B", "#B8860B", "#A9A9A9", "#006400", "#BDB76B", "#8B008B", "#556B2F"
        ]
    # Default: extended Tableau-like palette (20)
    return [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
        "#edc949","#af7aa1","#ff9da7","#9c755f","#bab0ab"
    ]

def build_unique_node_colors(processed_dict_single_subreddit, palette_name: str = "distinct") -> dict:
    """
    Build completely unique colors for all communities across all timesteps.
    No lineage consideration - every community gets a distinct color.
    """
    all_comm_keys = []
    timesteps = sorted(processed_dict_single_subreddit.keys())
    
    # Collect all community keys
    for ts in timesteps:
        comm_nodes = processed_dict_single_subreddit[ts]['community_info']['comm_nodes']
        for cid in comm_nodes.keys():
            all_comm_keys.append(f"{ts}_{cid}")
    
    n_total = len(all_comm_keys)
    print(f"Generating {n_total} unique colors for communities")
    
    if palette_name == "distinct":
        # Generate maximally distinct colors using golden angle
        colors = _generate_distinct_colors(n_total, saturation=0.8, lightness=0.6)
    elif palette_name == "random":
        # Generate random colors with good separation
        random.seed(42)  # For reproducibility
        colors = []
        for i in range(n_total):
            # Random hue, fixed saturation and lightness for consistency
            hue = random.random()
            r, g, b = colorsys.hls_to_rgb(hue, 0.6, 0.8)
            hex_color = '#{:02X}{:02X}{:02X}'.format(
                int(round(r * 255)), 
                int(round(g * 255)), 
                int(round(b * 255))
            )
            colors.append(hex_color)
    else:
        # Fall back to extended glasbey + generated colors
        base_palette = _get_palette("glasbey32")
        if n_total <= len(base_palette):
            colors = base_palette[:n_total]
        else:
            extra_needed = n_total - len(base_palette)
            extra_colors = _generate_distinct_colors(extra_needed, saturation=0.7, lightness=0.5)
            colors = base_palette + extra_colors
    
    # Shuffle colors to avoid any systematic patterns
    color_indices = list(range(len(colors)))
    random.seed(123)  # Different seed for shuffling
    random.shuffle(color_indices)
    
    # Assign shuffled colors to communities
    color_map = {}
    for i, comm_key in enumerate(all_comm_keys):
        color_map[comm_key] = colors[color_indices[i]]
    
    print(f"Sample colors: {list(colors[:5])}")
    return color_map

# Also improve the extra color generation for lineage mode
def _gen_extra_colors(n: int):
    """Generate additional distinct colors when base palette is exhausted"""
    return _generate_distinct_colors(n, saturation=0.75, lightness=0.55)

def build_lineage_colors(processed_dict_single_subreddit,
                         evolution_data,
                         mode: str = "hungarian",
                         min_jaccard_parent: float = 0.05,
                         base_palette=None,
                         split_strategy: str = "new_hues"
                         ) -> dict:
    """
    Build lineage-aware color mapping for communities.
    Ensures no two communities in the same timestep share colors.
    """
    if base_palette is None:
        base_palette = _get_palette("glasbey32")

    timesteps = sorted(processed_dict_single_subreddit.keys())
    if not timesteps:
        return {}

    # Debug: Check evolution_data structure
    # print(f"Debug: evolution_data keys: {list(evolution_data.keys()) if isinstance(evolution_data, dict) else 'Not a dict'}")
    if 'hungarian_matches' in evolution_data:
        hungarian_matches = evolution_data['hungarian_matches']
        # print(f"Debug: hungarian_matches type: {type(hungarian_matches)}")
        # if isinstance(hungarian_matches, list) and len(hungarian_matches) > 0:
        #     print(f"Debug: first hungarian match: {type(hungarian_matches[0])}, keys: {hungarian_matches[0].keys() if isinstance(hungarian_matches[0], dict) else 'Not a dict'}")

    # Expand palette if needed
    max_comms_per_timestep = max(
        len(processed_dict_single_subreddit[ts]['community_info']['comm_nodes']) 
        for ts in timesteps
    )
    
    needed_colors = max(max_comms_per_timestep, len(base_palette))
    if needed_colors > len(base_palette):
        extra_colors = _generate_distinct_colors(needed_colors - len(base_palette), 
                                                saturation=0.8, lightness=0.6)
        extended_palette = base_palette + extra_colors
    else:
        extended_palette = base_palette

    print(f"Using {len(extended_palette)} colors for lineage mode")
    print(f"Max communities per timestep: {max_comms_per_timestep}")

    # Initialize structures
    color_map = {}
    lineage_seed = {}  # (ts, comm) -> base_hex
    timestep_used_colors = {}  # ts -> set of colors used in that timestep

    # Assign seed colors at first timestep by community size (stable ordering)
    first_ts = timesteps[0]
    comm_nodes_t0 = processed_dict_single_subreddit[first_ts]['community_info']['comm_nodes']
    t0_order = sorted(comm_nodes_t0.keys(), key=lambda c: -len(comm_nodes_t0[c]))
    
    timestep_used_colors[first_ts] = set()
    next_color_idx = 0

    for cid in t0_order:
        base_col = extended_palette[next_color_idx % len(extended_palette)]
        lineage_seed[(first_ts, cid)] = base_col
        color_map[f"{first_ts}_{cid}"] = base_col
        timestep_used_colors[first_ts].add(base_col)
        next_color_idx += 1

    # Walk forward in time
    hungarian_matches = evolution_data.get('hungarian_matches', [])

    for idx in range(len(timesteps)-1):
        t1, t2 = timesteps[idx], timesteps[idx+1]
        comms_t1 = processed_dict_single_subreddit[t1]['community_info']['comm_nodes']
        comms_t2 = processed_dict_single_subreddit[t2]['community_info']['comm_nodes']
        
        timestep_used_colors[t2] = set()  # Track colors used in this timestep

        # Extract Jaccard matrix for this transition - fix the data structure access
        jac_matrix = None
        assignments = []
        
        if idx < len(hungarian_matches):
            hungarian_data = hungarian_matches[idx]
            if isinstance(hungarian_data, dict):
                # Check different possible keys
                if 'jaccard_matrix' in hungarian_data:
                    jac_matrix = hungarian_data['jaccard_matrix']
                elif 'matrix' in hungarian_data:
                    jac_matrix = hungarian_data['matrix']
                
                if 'assignments' in hungarian_data:
                    assignments = hungarian_data['assignments']
                elif 'assignment' in hungarian_data:
                    assignments = hungarian_data['assignment']
                elif 'matches' in hungarian_data:
                    assignments = hungarian_data['matches']

        # Find parent-child relationships
        parent_of_child = {}
        children_by_parent = {}

        if mode == "hungarian" and jac_matrix is not None and assignments:
            try:
                for row_idx, col_idx in assignments:
                    if (row_idx < len(list(comms_t1.keys())) and 
                        col_idx < len(list(comms_t2.keys()))):
                        jaccard_val = jac_matrix[row_idx, col_idx]
                        if jaccard_val >= min_jaccard_parent:
                            parent_comm = list(comms_t1.keys())[row_idx]
                            child_comm = list(comms_t2.keys())[col_idx]
                            parent_of_child[child_comm] = parent_comm
                            children_by_parent.setdefault(parent_comm, []).append(child_comm)
            except Exception as e:
                print(f"Warning: Error processing hungarian assignments for timestep {t1}->{t2}: {e}")
                print(f"jac_matrix shape: {jac_matrix.shape if hasattr(jac_matrix, 'shape') else 'no shape'}")
                print(f"assignments: {assignments}")

        # Handle orphaned children (no parent assigned)
        all_children = set(comms_t2.keys())
        assigned_children = set(parent_of_child.keys())
        orphaned_children = all_children - assigned_children
        if orphaned_children:
            children_by_parent[None] = list(orphaned_children)

        # Function to get next available color for this timestep
        def get_next_available_color():
            nonlocal next_color_idx
            attempts = 0
            while attempts < len(extended_palette) * 2:
                candidate_color = extended_palette[next_color_idx % len(extended_palette)]
                next_color_idx += 1
                # Check if this color is already used in the current timestep
                if candidate_color not in timestep_used_colors[t2]:
                    return candidate_color
                attempts += 1
            
            # If we've exhausted attempts, generate a new distinct color
            new_color = _generate_distinct_colors(1, saturation=0.8, lightness=0.6)[0]
            return new_color

        # Assign colors to children, ensuring uniqueness within timestep
        for parent, children in children_by_parent.items():
            children_sorted = sorted(children, key=lambda c: -len(comms_t2[c]))
            
            if parent is None:
                # New lineage; assign fresh colors to each child
                for child in children_sorted:
                    child_col = get_next_available_color()
                    lineage_seed[(t2, child)] = child_col
                    color_map[f"{t2}_{child}"] = child_col
                    timestep_used_colors[t2].add(child_col)
            else:
                parent_key = (t1, parent)
                parent_base_col = lineage_seed.get(parent_key)
                
                n = len(children_sorted)
                for idx_c, child in enumerate(children_sorted):
                    if split_strategy == "new_hues":
                        if idx_c == 0 and n == 1:
                            # Single child: try to inherit parent color if not used
                            if parent_base_col and parent_base_col not in timestep_used_colors[t2]:
                                child_col = parent_base_col
                                child_base = parent_base_col
                            else:
                                # Parent color already used, get new one
                                child_col = get_next_available_color()
                                child_base = child_col
                        else:
                            # Multiple children: each gets unique color
                            child_col = get_next_available_color()
                            child_base = child_col
                    else:
                        # Default: ensure uniqueness
                        child_col = get_next_available_color()
                        child_base = parent_base_col if parent_base_col else child_col
                    
                    lineage_seed[(t2, child)] = child_base
                    color_map[f"{t2}_{child}"] = child_col
                    timestep_used_colors[t2].add(child_col)

        # print(f"Timestep {t2}: {len(timestep_used_colors[t2])} unique colors assigned to {len(comms_t2)} communities")

    return color_map