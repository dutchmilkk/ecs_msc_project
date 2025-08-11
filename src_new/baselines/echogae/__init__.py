"""
EchoGAE baseline implementation for echo chamber analysis.

Based on:
Alatawi, F., Sheth, P., & Liu, H. "Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach."
In Proceedings of ASONAM '23: International Conference on Advances in Social Networks Analysis and Mining, 2023.
"""

from .GAE import GCNEncoder
from .EchoGAE import EchoGAE_algorithm
from .echo_chamber_measure import EchoChamberMeasure
from .RWC_jit import RWC

__all__ = ['EchoGAE_algorithm', 'EchoChamberMeasure', 'RWC', 'GCNEncoder']
