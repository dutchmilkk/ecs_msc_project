import numpy as np
from typing import Dict, Tuple

def compute_edge_weight(edge_data: Dict, strategy: str, scale: Tuple[float, float] = (1.0, 2.0)) -> float:
    """
    Compute a scalar edge weight from edge attributes using several strategies.

    Expected edge_data keys:
      net_vector: iterable of length 3 -> (disagree, neutral, agree) either counts or proportions (will be normalized if drift)
      confidence (optional): float in [0,1] (if missing defaults to 0.0)

    Parameters:
      strategy:
        - uniform: constant 1.0
        - agreement_diff: scales alignment ((agree - disagree + 1)/2) -> near 1 when agree >> disagree
        - confidence_sqrt: emphasizes low confidence more (sqrt(conf))
        - polarization: (agree + disagree) i.e. non-neutral mass
        - low_neutrality: 1 - neutral (high when neutral mass is small)
        - echo_intensity: alignment * polarization * sqrt(conf)
      scale: (lo, hi) numeric clamp range used by _scale01 (values are clipped into [lo, hi])

    Returns:
      float edge weight
    """
    lo, hi = scale
    def _scale01(value: float) -> float:
        # Clamp value into desired [lo, hi] interval
        return np.clip(value, lo, hi)

    # 1. Extract and normalize (if needed) disagree/neutral/agree triple from net_vector
    net_vec = edge_data.get('net_vector')
    disagree = neutral = agree = 0.0
    if isinstance(net_vec, (list, tuple, np.ndarray)) and len(net_vec) == 3:
        disagree, neutral, agree = map(float, net_vec)
        total = disagree + neutral + agree
        if total > 0 and abs(total - 1.0) > 1e-6:
            disagree /= total
            neutral /= total
            agree /= total

    # 2. Extract confidence and clip to [0, 1]
    conf = float(edge_data.get('confidence', 0.0))
    conf = max(0.0, min(1.0, conf))

    # 3. Select strategy
    if strategy == 'uniform':
        return 1.0
    if strategy == 'agreement_diff':
        # Alignment centered: (agree - disagree) in [-1,1] -> map to [0,1]
        return _scale01((agree - disagree + 1.0)/2.0)
    if strategy == 'confidence_sqrt':
        return _scale01(np.sqrt(conf))
    if strategy == 'polarization':
        return _scale01(agree + disagree)
    if strategy == 'low_neutrality':
        # Non-neutral share
        return _scale01(1.0 - neutral)
    if strategy == 'echo_intensity':
        # Joint effect: alignment * polarization * sqrt(conf)
        alignment = (agree - disagree + 1.0)/2.0
        polarization = (agree + disagree)
        return _scale01(alignment * polarization * (conf ** 0.5))
    raise ValueError(f"Unknown weight strategy: {strategy}")