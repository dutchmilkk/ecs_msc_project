"""
This implementation is based on:
Alatawi, F., Sheth, P., & Liu, H. "Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach."
In Proceedings of ASONAM '23: International Conference on Advances in Social Networks Analysis and Mining, 2023.

DOI: 10.1145/3625007.3627731
GitHub: https://github.com/faalatawi/echo-chamber-score/blob/main/src/echo_chamber_measure.py
"""

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import numpy as np
import logging

class EchoChamberMeasure:
    def __init__(
        self,
        users_representations: np.ndarray,
        labels: np.ndarray,
        metric: str = "euclidean",
    ):
        if metric == "euclidean":
            self.distances = euclidean_distances(users_representations)
        elif metric == "cosine":
            self.distances = cosine_distances(users_representations)
        self.labels = labels

        # [2025.07.26 - VD] Add logging setup
        self.logger = logging.getLogger(__name__)
        if self.logger.handlers:
            self.logger.handlers.clear()
        self.logger.propagate = True

    def cohesion_node(self, idx: int) -> float:
        """Average distance of a node to all other nodes in the same community."""
        node_label = self.labels[idx]

        node_distances = self.distances[idx, self.labels == node_label]

        return np.mean(node_distances)

    def separation_node(self, idx: int) -> float:
        """Minimum average distance to any other community."""
        node_label = self.labels[idx]

        dist = []
        for l in np.unique(self.labels):
            if l == node_label:
                continue
            dist.append(np.mean(self.distances[idx, self.labels == l]))

        return np.min(dist)

    def metric(self, idx: int) -> float:
        """Combines cohesion and separation metrics"""
        a = self.cohesion_node(idx)

        # [2025.07.26 - VD] Add error handling for separation_node
        try:
            b = self.separation_node(idx)
        except ValueError:
            b = 0.0
            self.logger.warning(f"separation_node() failed for index {idx}, returning 0.0")
        
        # [2025.07.26 - VD] Add error handling for division by zero
        denom = 2 * max(a, b)
        if denom == 0:
            self.logger.warning(f"Both cohesion and separation are zero for index {idx}, returning 0.0")
            return 0.0

        return (-a + b + max(a, b)) / denom if denom != 0 else 0.0

    def echo_chamber_index(self) -> float:
        """Average echo chamber index across all nodes."""
        nodes_metric = []
        for i in range(self.distances.shape[0]):
            nodes_metric.append(self.metric(i))
        return np.mean(nodes_metric)

    def community_echo_chamber_index(self, community_label: int) -> float:
        """Average echo chamber index for a specific community."""
        com_eci = []

        for i in range(self.distances.shape[0]):
            if self.labels[i] == community_label:
                com_eci.append(self.metric(i))

        return np.mean(com_eci)
