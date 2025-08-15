"""
Multitask DebateGNN for Online Discussion Analysis

Implements a Graph Neural Network architecture for analyzing online debate discussions.
The model performs multitask learning to jointly predict:
1. Link prediction (connections between users)  
2. Confidence prediction (reliability/certainty of relationships)
3. Net (dis)agreement vector prediction (soft probabilities of disagree/neutral/agree between users)

Key Components:
- ECCConv: Edge-Conditioned Convolution layer that uses edge attributes (confidence + net (dis)agreement vector)
- MultitaskDebateGNN: Main model architecture with multiple prediction heads
- Training pipeline with cross-validation across subreddits
- Uncertainty quantification through learnable log-variance parameters

The model is designed for temporal graph data where nodes represent users and edges
represent aggregated reply interactions. Edge attributes include:
- Confidence: Computed from annotator agreement fraction × individual kappa scores
- Net (dis)agreement vector: [disagree, neutral, agree] weighted by stance confidence scores (sums to 1)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from copy import deepcopy
import numpy as np
import random
import warnings
from collections import defaultdict

from IPython.display import clear_output
import matplotlib.pyplot as plt

from typing import Sequence, Mapping, Any, Optional
import inspect

# ============================================================================
# ECC LAYER
# ============================================================================
class ECCConv(MessagePassing):
    """
    Edge-Conditioned Convolution (ECC) Layer for Debate Analysis
    Adapted from "Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs"
    (Simonovsky et al., 2017). https://doi.org/10.48550/arXiv.1704.02901
    
    A custom PyTorch Geometric MessagePassing layer that conditions message computation
    on edge attributes (confidence and net (dis)agreement information). The layer uses an edge MLP
    to generate per-edge transformation matrices based on edge attributes.
    
    Architecture:
    1. Encodes net (dis)agreement vectors through a small MLP
    2. Concatenates confidence and encoded net (dis)agreement features
    3. Uses edge MLP to generate transformation weights for each edge
    4. Applies edge-specific transformations during message passing
    5. Includes residual connection and dropout for training stability
    
    Args:
        in_channels (int): Number of input node features
        out_channels (int): Number of output node features  
        hidden_dim (int, optional): Hidden dimension for edge MLP. Defaults to 128.
        conf_dim (int, optional): Dimension of confidence features. Defaults to 1.
        stance_dim (int, optional): Dimension of net (dis)agreement features. Defaults to 3.
        edge_mlp_dropout (float, optional): Dropout rate in edge MLP. Defaults to 0.1.
        keep_prob (float, optional): Probability of keeping edges during training (DropEdge). Defaults to 1.0.
    
    Input edge_attr format: 
        [confidence (conf_dim), net_disagreement_vector (stance_dim), ...]
        - Confidence: Annotator agreement fraction × individual kappa scores  
        - Net (dis)agreement vector: [disagree, neutral, agree] weighted by stance confidence (sums to 1)
    """
    def __init__(self, in_channels, out_channels, hidden_dim=128,
                 conf_dim=1, stance_dim=3, 
                 edge_mlp_dropout=0.1, keep_prob=1.0):
        super().__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conf_dim = conf_dim
        self.stance_dim = stance_dim
        self.keep_prob = keep_prob

        # Stance vector encoder
        self.vector_head = nn.Sequential(
            nn.Linear(stance_dim, 16),
            nn.GELU()
        )

        # Edge-conditioned MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(conf_dim + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(edge_mlp_dropout),
            nn.Linear(hidden_dim, in_channels * out_channels)
        )
        
        self.root_lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layer parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.vector_head[0].weight)
        nn.init.zeros_(self.vector_head[0].bias)
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.root_lin.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ECC layer.
        
        Args:
            x (torch.Tensor): Node features [N, in_channels]
            edge_index (torch.Tensor): Edge connectivity [2, E] 
            edge_attr (torch.Tensor): Edge attributes [E, conf_dim + stance_dim]
                Format: [confidence, net_disagreement_vector]
                - Confidence: Annotator agreement fraction × individual kappa scores
                - Net disagreement: [disagree, neutral, agree] weighted by stance confidence (sums to 1)
        
        Returns:
            torch.Tensor: Updated node features [N, out_channels]
        """
        if self.training and self.keep_prob < 1.0:
            mask = torch.rand(edge_attr.size(0), device=edge_attr.device) < self.keep_prob
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]
        
        confidence = edge_attr[:, :self.conf_dim]
        stance_vec = edge_attr[:, self.conf_dim:self.conf_dim + self.stance_dim]
        stance_emb = self.vector_head(stance_vec)
        edge_input = torch.cat([confidence, stance_emb], dim=1)

        if torch.isnan(edge_input).any():
            raise ValueError("NaN detected in edge_input")
            
        out = self.propagate(edge_index, x=x, edge_input=edge_input)
        out = F.gelu(out + self.root_lin(x) + self.bias)
        return out
    
    def message(self, x_j: torch.Tensor, edge_input: torch.Tensor) -> torch.Tensor: # type: ignore        
        """
        Compute messages for each edge using edge-conditioned transformations.
        
        This method generates a unique transformation matrix for each edge based on
        the edge attributes, then applies it to the source node features.
        
        Args:
            x_j (torch.Tensor): Source node features [E, in_channels]
            edge_input (torch.Tensor): Processed edge features [E, conf_dim + 16] 
        
        Returns:
            torch.Tensor: Transformed messages [E, out_channels]
        """
        w = self.edge_mlp(edge_input)
        w = w.view(-1, self.in_channels, self.out_channels)
        w = w / torch.norm(w, dim=(1,2), keepdim=True).clamp(min=1e-6)
        
        x_j = x_j.unsqueeze(1)
        out = torch.bmm(x_j, w).squeeze(1)
        if torch.isnan(out).any():
            raise ValueError("NaN in message output")

        return out

# ============================================================================
# MULTITASK DEBATE GNN - Model Architecture
# ============================================================================
class MultitaskDebateGNN(nn.Module):
    """
    Multitask Graph Neural Network for Online Debate Analysis
    
    A GNN architecture that jointly learns three tasks on debate discussion graphs:
    1. Link Prediction: Predicting connections between users
    2. Confidence Prediction: Estimating reliability/certainty of relationships  
    3. Net (dis)agreement Vector Prediction: Predicting soft probabilities (disagree/neutral/agree) between users
    
    Architecture Overview:
    - Stack of GNN layers (ECCConv, GCNConv, or SAGEConv)
    - Layer normalization and skip connections
    - Separate prediction heads for each task
    - Uncertainty quantification via learnable log-variance parameters
    - Multitask loss weighting with automatic balancing
    
    Training Features:
    - Supports multiple modes: "full", "no_stance", "no_conf", "link_only"
    - Temporal regularization for sequential graphs
    - Label smoothing and dropout regularization
    - Class-balanced net (dis)agreement loss with prior knowledge
    
    Edge Attribute Details:
    - Confidence: Computed from annotator agreement fraction × individual kappa scores
    - Net (dis)agreement vector: [disagree, neutral, agree] weighted by stance confidence scores (sums to 1)
    
    Args:
        in_dim (int): Input node feature dimension
        hidden_dim (int): Hidden dimension for GNN layers
        emb_dim (int): Final embedding dimension
        conf_dim (int, optional): Confidence feature dimension. Defaults to 1.
        stance_dim (int, optional): Net (dis)agreement feature dimension. Defaults to 3.
        neg_attr_weight (float, optional): Weight for negative edges in attribute prediction. Defaults to 0.25.
        dropout (float, optional): General dropout rate. Defaults to 0.3.
        num_layers (int, optional): Number of GNN layers. Defaults to 3.
        mode (str, optional): Training mode. Defaults to "full".
        conv_cls (type, optional): Single convolution class for all layers. Defaults to None.
        conv_cls_list (Sequence[type], optional): List of convolution classes per layer. Defaults to None.
        ecc_kwargs (Mapping[str, Any], optional): Additional arguments for ECC layers. Defaults to None.
    """
    def __init__(self, in_dim, hidden_dim, emb_dim,
                 conf_dim: int = 1, stance_dim: int = 3,
                 *,
                 neg_attr_weight: float = 0.25,
                 dropout: float = 0.3,
                 num_layers: int = 3,
                 mode: str = "full",
                 conv_cls: type | None = None,
                 conv_cls_list: Sequence[type] | None = None,
                 ecc_kwargs: Mapping[str, Any] | None = None):
        super().__init__()

        self.stance_dim = stance_dim
        self.conf_dim = conf_dim
        self.mode = mode
        self.use_conf = mode in ("full", "no_stance")
        self.use_stance = mode in ("full", "no_conf")

        self.neg_attr_weight = neg_attr_weight

        ecc_kwargs = dict(ecc_kwargs or {})

        if conv_cls_list is None:
            if conv_cls is None:
                from __main__ import ECCConv
                conv_cls_list = [ECCConv] * num_layers
            else:
                conv_cls_list = [conv_cls] * num_layers      
            self.is_hybrid  = False
            self.conv_types = conv_cls_list      
        else:
            if len(conv_cls_list) != num_layers:
                raise ValueError(f"conv_cls_list must have length {num_layers}, got {len(conv_cls_list)}")
            self.is_hybrid  = True
            self.conv_types = list(conv_cls_list)

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [emb_dim]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i, c_cls in enumerate(conv_cls_list):
            in_channels, out_channels = dims[i], dims[i+1]
            if c_cls.__name__ == "ECCConv":
                sig_keys = inspect.signature(c_cls).parameters.keys()
                layer_kwargs = {k: v for k, v in ecc_kwargs.items() if k in sig_keys}
                conv = c_cls(in_channels, out_channels, **layer_kwargs)
            else:
                conv = c_cls(in_channels, out_channels)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(out_channels))
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, emb_dim) if in_dim != emb_dim else nn.Identity()

        self.link_head = nn.Sequential(
            nn.Linear(2 * emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.edge_head = nn.Sequential(
            nn.Linear(2 * emb_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        ) if mode != "link_only" else None

        self.log_var_link = nn.Parameter(torch.zeros(1))
        self.log_var_conf = nn.Parameter(torch.zeros(1)) if self.use_conf else None
        self.log_var_stance = nn.Parameter(torch.zeros(1)) if self.use_stance else None
        
        # Initialize weights safely
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize prediction head parameters with small gains for stability."""
        for layer in self.link_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
        if self.edge_head is not None:
            for layer in self.edge_head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
                    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the GNN to compute node embeddings.
        
        Args:
            x (torch.Tensor): Input node features [N, in_dim]
            edge_index (torch.Tensor): Edge connectivity [2, E]
            edge_attr (torch.Tensor): Edge attributes [E, attr_dim] (if applicable)
        
        Returns:
            torch.Tensor: Node embeddings [N, emb_dim]
        """
        residual = self.skip(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if hasattr(conv, "edge_mlp"):
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            
            x = norm(x)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        return self.activation(residual + x)
    
    @torch.no_grad()
    def embed(self, data: Data, 
              device: str | torch.device | None = None,
              eval_mode: bool = True) -> torch.Tensor:
        """
        Generate node embeddings for a graph in evaluation mode.
        
        Args:
            data (Data): PyTorch Geometric data object
            device (str | torch.device | None, optional): Target device. Defaults to None.
            eval_mode (bool, optional): Whether to use evaluation mode. Defaults to True.
        
        Returns:
            torch.Tensor: Node embeddings [N, emb_dim]
        """
        return get_node_embeddings(self, data, device, eval_mode)
    
    def predict_link(self, z, edge_index) -> torch.Tensor:
        """
        Predict link probabilities between node pairs.
        
        Args:
            z (torch.Tensor): Node embeddings [N, emb_dim]
            edge_index (torch.Tensor): Edge pairs to predict [2, E]
        
        Returns:
            torch.Tensor: Link prediction logits [E]
        """
        src, dst = edge_index
        logits = self.link_head(torch.cat([z[src], z[dst]], dim=1)).squeeze(1)
        return logits
    
    def predict_edge_attr(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor | None:
        """
        Predict edge attributes (confidence + net (dis)agreement vector).
        
        Combines node embeddings to predict edge-level attributes:
        - Confidence: Sigmoid activation for binary confidence prediction 
            (targets: annotator agreement fraction × individual kappa scores)
        - Net (dis)agreement vector: Softmax over 3 values (disagree/neutral/agree)
            (targets: soft probabilities weighted by stance confidence scores, sum to 1)
        
        Args:
            z (torch.Tensor): Node embeddings [N, emb_dim]
            edge_index (torch.Tensor): Edge indices [2, E]
        
        Returns:
            torch.Tensor | None: Edge attribute predictions [E, conf_dim + stance_dim] or None if no edge head
        """
        if self.edge_head is None or not self.use_conf and not self.use_stance:
            return None
    
        src, dst = edge_index
        edge_input = torch.cat([z[src], z[dst]], dim=1)
        
        # Add stability checks
        if torch.isnan(edge_input).any():
            raise ValueError("NaN detected in edge prediction input")
        
        raw = self.edge_head(edge_input)
        
        # Process outputs with stability safeguards
        results = []
        if self.use_conf:
            conf = torch.sigmoid(raw[:, 0].unsqueeze(1))  # shape [E, 1]
            results.append(conf)
        
        if self.use_stance:
            start_idx = 1 if self.use_conf else 0
            stance_logits = raw[:, start_idx:start_idx + self.stance_dim]
            
            # Stable softmax
            stance_logits = stance_logits - stance_logits.max(dim=1, keepdim=True).values
            stance = F.softmax(stance_logits, dim=1).clamp(min=1e-6, max=1-1e-6)
            results.append(stance)
        
        if not results:
            return None
        
        return torch.cat(results, dim=1) if len(results) > 1 else results[0]
    
    def uncertainty_weighted_loss(self, link_loss, conf_loss, stance_loss, task_weights=None):
        """
        Compute uncertainty-weighted multitask loss using learnable log-variance parameters
        with optional custom task boosting.
        
        Implements the multitask uncertainty weighting from "Multi-Task Learning Using 
        Uncertainty to Weigh Losses for Scene Geometry and Semantics" (Kendall et al.)
        with additional custom task weighting support.
        
        Args:
            link_loss (torch.Tensor): Link prediction loss
            conf_loss (torch.Tensor): Confidence prediction loss
            stance_loss (torch.Tensor): Stance prediction loss
            task_weights (dict, optional): Custom task weights {"link": float, "conf": float, "stance": float}

        Returns:
            torch.Tensor: Weighted total loss
        """
        task_losses, log_vars, task_names = [], [], []
        total = torch.tensor(0.0, device=link_loss.device, dtype=link_loss.dtype)

        # Collect active tasks
        task_losses.append(link_loss)
        log_vars.append(self.log_var_link)
        task_names.append("link")
        
        if self.use_conf:
            task_losses.append(conf_loss)
            log_vars.append(self.log_var_conf)
            task_names.append("conf")
            
        if self.use_stance:
            task_losses.append(stance_loss)
            log_vars.append(self.log_var_stance)
            task_names.append("stance")
        
        # Apply uncertainty weighting + custom task boosting
        for loss, s, task_name in zip(task_losses, log_vars, task_names):
            # Base uncertainty weighting
            precision = torch.exp(-s).clamp(min=0.2, max=5.0)
            weighted_term = 0.5 * (loss * precision.squeeze() + s.squeeze())
            
            # Apply custom task boost if provided
            if task_weights is not None and task_name in task_weights:
                boost_factor = task_weights[task_name]
                weighted_term = weighted_term * boost_factor
            else:
                boost_factor = 1.0

            total += weighted_term
    
        return total
    # def uncertainty_weighted_loss(self, link_loss, conf_loss, stance_loss):
    #     """
    #     Compute uncertainty-weighted multitask loss using learnable log-variance parameters.
        
    #     Implements the multitask uncertainty weighting from "Multi-Task Learning Using 
    #     Uncertainty to Weigh Losses for Scene Geometry and Semantics" (Kendall et al.).
        
    #     Args:
    #         link_loss (torch.Tensor): Link prediction loss
    #         conf_loss (torch.Tensor): Confidence prediction loss
    #         stance_loss (torch.Tensor): Stance prediction loss

    #     Returns:
    #         torch.Tensor: Weighted total loss
    #     """
    #     task_losses, log_vars = [], []
    #     total = torch.tensor(0.0, device=link_loss.device, dtype=link_loss.dtype)

    #     task_losses.append(link_loss)
    #     log_vars.append(self.log_var_link)
    #     if self.use_conf:
    #         task_losses.append(conf_loss)
    #         log_vars.append(self.log_var_conf)
    #     if self.use_stance:
    #         task_losses.append(stance_loss)
    #         log_vars.append(self.log_var_stance)
        
    #     for loss, s in zip(task_losses, log_vars):
    #         precision = torch.exp(-s).clamp(min=0.2, max=5.0)
    #         weighted_term = 0.5 * (loss * precision.squeeze() + s.squeeze())
    #         total += weighted_term
      
    #     return total

    def compute_losses(self, z: torch.Tensor, edge_attr: torch.Tensor, 
                       pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor, 
                       pos_weight: torch.Tensor | None = None,
                       stance_weight: torch.Tensor | None = None,
                       task_weights: dict | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute multitask losses for link prediction, confidence, and net (dis)agreement vector prediction.
        
        Loss Components:
        1. Link Loss: Binary cross-entropy for edge existence prediction
        2. Confidence Loss: Smooth L1 loss for confidence prediction
           (targets: annotator agreement fraction × individual kappa scores)
        3. Net (dis)agreement Loss: KL divergence for net (dis)agreement vector prediction
           (targets: soft probabilities weighted by stance confidence scores, sum to 1)
        
        Key Features:
        - Handles positive/negative edge sampling imbalance
        - Uses class weights for net (dis)agreement balancing
        - Applies soft targets to negative edges
        
        Args:
            z (torch.Tensor): Node embeddings [N, emb_dim]
            edge_attr (torch.Tensor): Edge attributes for positive edges [E_pos, attr_dim]
            pos_edge_index (torch.Tensor): Positive edge indices [2, E_pos]
            neg_edge_index (torch.Tensor): Negative edge indices [2, E_neg]
            pos_weight (torch.Tensor | None, optional): Positive class weight for link prediction
            stance_weight (torch.Tensor | None, optional): Class weights for net (dis)agreement prediction
            task_weights (dict | None, optional): Custom task weights {"link": float, "conf": float, "stance": float}
        
        Returns:
            tuple: (total_loss, link_loss, conf_loss, stance_loss)
        """
        
        device = z.device

        pos_edge_index = pos_edge_index.to(device)
        neg_edge_index = neg_edge_index.to(device)
        edge_attr = edge_attr.to(device)
        stance_weight = stance_weight.to(device) if isinstance(stance_weight, torch.Tensor) else None
        if pos_weight is not None:
            pos_weight = torch.clamp(pos_weight, max=50.0)  # Prevent extreme values
        
        pos_logits = self.predict_link(z, pos_edge_index)
        neg_logits = self.predict_link(z, neg_edge_index)
        link_pred = torch.cat([pos_logits, neg_logits], dim=0)
        link_labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        if pos_weight is not None:
            link_loss = F.binary_cross_entropy_with_logits(link_pred, link_labels, pos_weight=pos_weight)
        else:
            link_loss = F.binary_cross_entropy_with_logits(link_pred, link_labels)
        
        conf_loss = stance_loss = torch.tensor(0.0, device=device)
        if self.edge_head is not None:
            edge_idx_all = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            edge_attr_pred = self.predict_edge_attr(z, edge_idx_all)

            if edge_attr_pred is not None:
                if edge_attr_pred.dim() == 1:
                    edge_attr_pred = edge_attr_pred.unsqueeze(1)
                
                E_pos = pos_edge_index.size(1)

                if self.use_conf:
                    true_conf_pos = edge_attr[:, 0] # 0/1 for real edges
                    true_conf_neg = torch.zeros(neg_edge_index.size(1), device=device)
                    conf_target = torch.cat([true_conf_pos, true_conf_neg])
                    # OPTIONAL: give negatives a *soft* zero instead of a hard one
                    conf_target[E_pos:] = conf_target[E_pos:] * (1 - self.neg_attr_weight)
        
                    pos_mask = torch.arange(len(conf_target), device=device) < pos_edge_index.size(1)
                    sample_w = torch.where(pos_mask, # weight = 1.0 on positives
                                           torch.full_like(conf_target, self.neg_attr_weight),
                                           torch.ones_like(conf_target))
                    # conf_loss = F.binary_cross_entropy(edge_attr_pred[:, 0], conf_target, weight=sample_w)
                    
                    smooth_l1_losses = nn.SmoothL1Loss()(edge_attr_pred[:,0], conf_target)
                    conf_loss = (smooth_l1_losses * sample_w).mean()
                    
                if self.use_stance:
                    # stance target
                    true_stance_pos = edge_attr[:, 1:] # [E_pos, 3]
                    if stance_weight is not None:
                        prior = (stance_weight / stance_weight.sum()).to(z)
                    else:
                        prior = torch.full((self.stance_dim,), 1.0 / self.stance_dim, device=z.device)
                    true_stance_neg = prior.expand(neg_edge_index.size(1), -1)
                    stance_target = torch.cat([true_stance_pos, true_stance_neg], dim=0)
        
                    # KL-div with per-sample weights
                    stance_pred = edge_attr_pred[:, -self.stance_dim:].clamp_min(1e-8).log()
                    pos_mask = torch.arange(len(stance_target), device=device) < pos_edge_index.size(1)
                    sample_w = torch.where(pos_mask,
                                              torch.ones_like(stance_target[:, 0]),
                                              torch.full_like(stance_target[:, 0], self.neg_attr_weight))
                    stance_kl = F.kl_div(stance_pred, stance_target, reduction='none').sum(1)
                    stance_loss = (sample_w * stance_kl).mean()
        
        total = self.uncertainty_weighted_loss(link_loss, conf_loss, stance_loss, task_weights)
        return total, link_loss, conf_loss, stance_loss
    
    def get_uncertainty(self):
        """
        Get current uncertainty estimates for each task.
        
        Returns:
            dict: Dictionary with uncertainty values for link, conf, and stance tasks
        """
        return {
            "link": torch.exp(self.log_var_link).item(),
            "conf": torch.exp(self.log_var_conf).item() if self.log_var_conf is not None else None,
            "stance": torch.exp(self.log_var_stance).item() if self.log_var_stance is not None else None
        }


# ============================================================================
# MULTITASK DEBATE GNN - Training
# ============================================================================
def train_gnn_live(all_graphs, model_args, train_args, model_class=MultitaskDebateGNN, live_plot=True):
    """
    Train MultitaskDebateGNN with cross-validation across subreddits:
    
    - Cross-validation: Each subreddit serves as test set once
    - Three-way split: Train/validation/test using different subreddits
    - Live monitoring: Real-time loss plotting during training
    - Early stopping: Based on validation loss with patience
    - Temporal regularization: Consistency across time for sequential graphs
    - Class balancing: Automatic pos_weight and net (dis)agreement weight computation
    
    Training Strategy:
    - For each subreddit as test set:
      1. Select another random subreddit as validation
      2. Use remaining subreddits for training
      3. Train model with early stopping on validation loss
      4. Evaluate final model on held-out test subreddit
    
    Edge Attributes:
    - Confidence: Computed from annotator agreement fraction × individual kappa scores
    - Net (dis)agreement vector: [disagree, neutral, agree] weighted by stance confidence scores (sum to 1)
    
    Args:
        all_graphs (list): List of PyTorch Geometric Data objects with subreddit_id and local_timestep
        model_args (dict): Arguments for model initialization
        train_args (dict): Training configuration including lr, epochs, patience, etc.
        model_class (type, optional): Model class to instantiate. Defaults to MultitaskDebateGNN.
    
    Returns:
        tuple: (final_model, test_results_dict, training_history_dict)
        - final_model: Trained model from last fold
        - test_results_dict: Per-subreddit test metrics
        - training_history_dict: Per-subreddit training/validation loss curves
    
    Note:
        Graphs represent directed user-user interactions from aggregated reply relationships.
        Nodes are users, edges are aggregated reply interactions with confidence and net (dis)agreement attributes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack training arguments
    temp_reg_weight = train_args.get('temp_reg_weight', 0.0)
    neg_sample_ratio = train_args.get('neg_sample_ratio', 1.0)
    min_delta = train_args.get("min_delta", 0.0)
    patience = train_args.get("patience", 10)
    mode = model_args.get("mode", "full")
    task_weights = train_args.get("task_weights", None)  # Optional task weighting

    #===========================================
    # Dataset Overview
    #===========================================
    print(f"\nDataset Overview:")
    print(f"  - Total graphs: {len(all_graphs)}")
    print(f"  - Device: {device}")
    print(f"  - Training mode: {mode}")
    print(f"  - Task weights: {task_weights if task_weights else 'Default (1.0 each)'}")
    print(f"  - Tasks: {', '.join(['Link'] + (['Confidence'] if mode in ('full', 'no_stance') else []) + (['Stance'] if mode in ('full', 'no_conf') else []))}")

    # Organize graphs by subreddit
    subreddit_graphs = defaultdict(list)
    for g in all_graphs:
        if hasattr(g, "_prev_emb"): 
            delattr(g, "_prev_emb")
        subreddit_graphs[g.subreddit_id].append(g)
    
    # Sort each subreddit's graphs by time
    total_nodes = total_edges = 0
    for sid in subreddit_graphs:
        subreddit_graphs[sid] = sorted(subreddit_graphs[sid], key=lambda g: g.local_timestep)
        graphs = subreddit_graphs[sid]  # Get graphs for THIS subreddit
        
        # Calculate stats for this subreddit
        subreddit_nodes = sum(g.num_nodes for g in graphs)
        subreddit_edges = sum(g.edge_index.size(1) for g in graphs)
        avg_edges = sum(g.edge_index.size(1) for g in graphs) / len(graphs)
        stance_dist = torch.cat([g.edge_attr[:, 1:] for g in graphs]).mean(0).tolist()
        
        total_nodes += subreddit_nodes
        total_edges += subreddit_edges
        
        print(f"   + Subreddit {sid}: {len(graphs)} graphs | " f"{subreddit_nodes:,} nodes | {subreddit_edges:,} edges | " f"Avg: {avg_edges:.1f} edges/graph")
        print(f"   + Stance Dist: [{stance_dist[0]:.3f}, {stance_dist[1]:.3f}, {stance_dist[2]:.3f}] " f"(disagree/neutral/agree)")
    
    print("\nGlobal Statistics:")
    print(f"  - Total nodes: {total_nodes:,}")
    print(f"  - Total edges: {total_edges:,}")
    print(f"  - Avg nodes/graph: {total_nodes / len(all_graphs):.1f}")
    print(f"  - Avg edges/graph: {total_edges / len(all_graphs):.1f}")

    subreddit_ids = list(subreddit_graphs.keys())
    results = {}
    cv_history = {}
    model = None

    # ============================================================================
    # Edge Attribute Analysis
    # ============================================================================
    def analyze_edge_attributes(graphs, dataset_name="Dataset"):
        """Check correlations between confidence and stance classes."""
        conf = torch.cat([g.edge_attr[:, 0] for g in graphs])
        stance = torch.cat([g.edge_attr[:, 1:] for g in graphs])

        print(f"\nEdge Attribute ({dataset_name}):")
        print(f"  - Confidence: μ={conf.mean():.3f}, σ={conf.std():.3f}")
        print(f"  - Stance dist: [{stance.mean(0)[0]:.3f}, {stance.mean(0)[1]:.3f}, {stance.mean(0)[2]:.3f}] "
              f"(disagree/neutral/agree)")
        
        # Correlation analysis
        corrs = []
        for i, label in enumerate(['disagree', 'neutral', 'agree']):
            r = torch.corrcoef(torch.stack([conf, stance[:, i]]))[0, 1].item()
            corrs.append(f"{label}:{r:+.3f}")
        print(f"  - Confidence correlations: {', '.join(corrs)}")
    
    print(f"\nGlobal Edge Attribute Analysis:")
    analyze_edge_attributes(all_graphs, "Full Dataset")

    for test_sid in sorted(subreddit_graphs):
        print(f"\n{'='*50}")
        print(f"Training for Test Subreddit: {test_sid}")
        print(f"{'='*50}")
        
        # Three-way split (train/val/test)
        test_graphs = subreddit_graphs[test_sid]
        train_val_sids = [sid for sid in subreddit_ids if sid != test_sid]
        random.seed(42)
        val_sid = random.choice(train_val_sids)
        val_graphs = subreddit_graphs[val_sid]
        train_sids = [sid for sid in train_val_sids if sid != val_sid]
        train_graphs = [g for sid in train_sids for g in subreddit_graphs[sid]]
        
        print(f"\nData Split:")
        print(f"  - Training: {len(train_sids)} subreddits → {len(train_graphs)} graphs")
        print(f"    Subreddits: {sorted(train_sids)}")
        print(f"  - Validation: Subreddit {val_sid} → {len(val_graphs)} graphs")
        print(f"  - Testing:  Subreddit {test_sid} → {len(test_graphs)} graphs")

        # Analyze data splits
        analyze_edge_attributes(train_graphs, "Training")
        analyze_edge_attributes(val_graphs, f"Validation (Sub {val_sid})")
        
        # Compute weights from training data only
        stance_weight = compute_global_stance_weight(train_graphs)
        total_pos = total_possible_neg = 0
        for g in train_graphs:
            num_nodes = g.num_nodes
            num_edges = g.edge_index.size(1)
            total_pos += num_edges
            total_possible_neg += num_nodes * num_nodes - num_edges
        pos_weight = torch.tensor(
            min((total_possible_neg / total_pos) * 0.5, 50.0),  # Limit to prevent extreme weights
            device=device
        )
        print(f"\nLoss Balancing:")
        print(f"  - Link pos_weight: {pos_weight.item():.3f} (pos/neg ratio: {total_pos/total_possible_neg:.4f})")
        print(f"  - Stance weights:  [{stance_weight[0]:.3f}, {stance_weight[1]:.3f}, {stance_weight[2]:.3f}]")

        # Initialize model and optimizer
        model = model_class(**model_args).to(device)
        
        log_params = [p for n, p in model.named_parameters() if "log_var_" in n]
        base_params = [p for n, p in model.named_parameters() if "log_var_" not in n]

        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': train_args["lr"] * 0.1},  # Lower base LR
            {'params': log_params, 'lr': train_args["lr"] * 0.01}   # Very low for log_vars
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=1, eta_min=train_args["lr"] * 0.01
        )

        print("\nModel Configs:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,} (trainable: {trainable_params:,})")
        print(f"  - Base LR: {train_args['lr'] * 0.1:.2e}")
        print(f"  - Log-var LR: {train_args['lr'] * 0.01:.2e}")
        if task_weights:
            print(f"  - Custom task weights: {task_weights}")
        else:
            print(f"  - Task weights: Default (uncertainty-only)")

        # Training setup
        live = {
            "epoch": [], "train_total": [], "val_total": [], "train_link": [], "val_link": [],
        }
        if mode in ("full", "no_stance"):
            live["train_conf"] = live["val_conf"] = []
        if mode in ("full", "no_conf"):
            live["train_stance"] = live["val_stance"] = []
        
        # Training loop
        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0
        hist = defaultdict(list)
        
        print(f"\n*** BEGIN TRAINING ***")
        print(f"  - Max epochs: {train_args['epochs']}")
        print(f"  - Early stopping patience: {patience} epochs")
        print(f"  - Negative sampling ratio: {neg_sample_ratio:.2f}")
        if temp_reg_weight > 0.0:
            print(f"  - Temporal regularization weight: {temp_reg_weight:.3f}")
        
        # Training loop
        for epoch in range(train_args["epochs"]):
            model.train()
            n_batches = 0
            total = link = conf = stance = 0.0
            
            for batch_idx, g in enumerate(train_graphs):
                g = g.to(device)
                z = model(g.x, g.edge_index, g.edge_attr if mode != "link_only" else None)
                neg = negative_sampling(
                    g.edge_index, g.x.size(0), int(g.edge_index.size(1) * neg_sample_ratio)
                ).to(device)

                # Compute losses
                loss_total, loss_link, loss_conf, loss_stance = model.compute_losses(
                    z=z, 
                    edge_attr=g.edge_attr, 
                    pos_edge_index=g.edge_index, 
                    neg_edge_index=neg,
                    pos_weight=pos_weight,
                    stance_weight=stance_weight,
                    task_weights=task_weights
                )

                # Temporal regularization
                if temp_reg_weight > 0.0:
                    prev = getattr(g, "_prev_emb", None)
                    if prev is not None and hasattr(g, "node_map"):
                        first_vec = next(iter(prev.values()))
                        if first_vec.shape[0] == z.size(1):
                            common = set(prev) & set(g.node_map)
                            if common:
                                curr_idx = torch.tensor([g.node_map[u] for u in common], device=device)
                                prev_emb = torch.stack([prev[u] for u in common], dim=0).to(device)
                                temp_loss = F.mse_loss(z[curr_idx], prev_emb)
                                loss_total += temp_reg_weight * temp_loss
                
                # Store current embeddings for next timestep
                if hasattr(g, "node_map"):
                    g._prev_emb = {u: z[idx].detach().cpu() for u, idx in g.node_map.items()}
                
                optimizer.zero_grad()
                loss_total.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Monitor uncertainty parameters
                if hasattr(model, "log_var_stance") and model.log_var_stance is not None:
                    with torch.no_grad():
                        model.log_var_stance.clamp_(min=-2.5, max=2.5)
                
                # Accumulate losses
                total += loss_total.item()
                link += loss_link.item()
                conf += loss_conf.item()
                stance += loss_stance.item()
                n_batches += 1

            scheduler.step(epoch)

            # Record training losses
            train_total = total / n_batches
            train_link = link / n_batches
            train_conf = conf / n_batches
            train_stance = stance / n_batches
            hist["epoch"].append(epoch)
            hist["train_total"].append(train_total)
            hist["train_link"].append(train_link)
            # ONLY store task losses if task is active
            if mode in ("full", "no_stance"):
                hist["train_conf"].append(train_conf)
            if mode in ("full", "no_conf"):
                hist["train_stance"].append(train_stance)

            # Validation
            val_total, val_link, val_conf, val_stance = validate_model(
                model, val_graphs, device, pos_weight, stance_weight, mode
            )
            hist["val_total"].append(val_total)
            hist["val_link"].append(val_link)
            if mode in ("full", "no_stance"):
                hist["val_conf"].append(val_conf)
            if mode in ("full", "no_conf"):
                hist["val_stance"].append(val_stance)
            
            # Live plotting
            live["epoch"].append(epoch)
            live["train_total"].append(train_total)
            live["val_total"].append(val_total)
            live["train_link"].append(train_link)
            live["val_link"].append(val_link)
            if "train_conf" in live:
                live["train_conf"].append(train_conf)
                live["val_conf"].append(val_conf)
            if "train_stance" in live:
                live["train_stance"].append(train_stance)
                live["val_stance"].append(val_stance)
            
            if epoch % 1 == 0 and live_plot:
                _live_plot(live, test_sid)
            
            # Detailed progress every N epochs
            if epoch % 20 == 0 or epoch == 0:
                print(f"\nEpoch {epoch:3d}:")
                print(f"  - Train: Total={train_total:.4f} | Link={train_link:.4f} | "
                      f"Conf={train_conf:.4f} | Stance={train_stance:.4f}")
                print(f"  - Val:   Total={val_total:.4f} | Link={val_link:.4f} | "
                        f"Conf={val_conf:.4f} | Stance={val_stance:.4f}")
                # Uncertainty estimates
                if hasattr(model, "get_uncertainty"):
                    uncertainties = model.get_uncertainty()
                    unc_str = " | ".join([f"{k.title()}={v:.3f}" if v is not None else f"{k.title()}=N/A" for k, v in uncertainties.items()])
                    print(f"  - Uncertainty: {unc_str}")
                    print(f"  - LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping check
            if val_total < best_val - min_delta:
                best_val = val_total
                best_state = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
            
            torch.cuda.empty_cache()
    
        # Load best model and evaluate on TEST set
        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            warnings.warn("No validation improvement; using final model")
        
        # Final evaluation on test set
        print(f"\nFinal Evaluation on Test Subreddit {test_sid}:")
        test_metrics = evaluate_model(model, test_graphs, device, pos_weight, mode)
        results[test_sid] = test_metrics
        cv_history[test_sid] = hist

        print(f"\nTest Metrics for {test_sid}:")
        for metric, value in test_metrics.items():
            if value is not None:
                if 'auc' in metric.lower():
                    print(f"   - {metric.replace('_', ' ').title()}: {value:.4f}")
                elif 'nll' in metric.lower() or 'kl' in metric.lower():
                    print(f"   - {metric.replace('_', ' ').title()}: {value:.4f}")
                elif 'mae' in metric.lower():
                    print(f"   - {metric.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"   - {metric.replace('_', ' ').title()}: {value:.4f}")

    # ====================
    # FINAL SUMMARY
    # ====================
    metrics = next(iter(results.values())).keys()
    avg_metrics = {}
    for m in metrics:
        values = [results[f][m] for f in results if results[f][m] is not None]
        if values:
            avg_metrics[m] = np.mean(values)
    
    print("\n*** Per-Fold Results ***")
    for fid in sorted(results.keys()):
        print(f"  Subreddit {fid}:")
        for metric, value in results[fid].items():
            if value is not None:
                print(f"     - {metric.replace('_', ' ').title():18}: {value:.4f}")

    print("\n*** Average Test Performance ***")
    for m in sorted(avg_metrics.keys()):
        avg_val = avg_metrics[m]
        metric_str = f"{m.replace('_', ' ').title()}: {avg_val:.4f}"
        print(f"  - {metric_str:20}: {avg_val:.4f}")
    
    
    return model, results, cv_history

# ============================================================================
# VALIDATION AND EVALUATION
# ============================================================================
def validate_model(model, val_graphs, device, pos_weight, stance_weight, mode: str = "full"):
    """
    Validate model on multiple graphs from validation set.
    
    Computes average losses across all validation graphs for early stopping.
    
    Args:
        model (MultitaskDebateGNN): Model to validate
        val_graphs (list): List of validation graphs
        device (torch.device): Computation device
        pos_weight (torch.Tensor): Positive weight for link prediction
        stance_weight (torch.Tensor): Class weights for net (dis)agreement prediction
        mode (str, optional): Training mode. Defaults to "full".
        epoch (int | None, optional): Current epoch for curriculum scheduling
    
    Returns:
        tuple: (total_loss, link_loss, conf_loss, stance_loss) averaged across graphs
    """
    model.eval()
    total_loss = total_link = total_conf = total_stance = 0.0
    n_graphs = len(val_graphs)
    
    if n_graphs == 0:
        return 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for g in val_graphs:
            g = g.to(device)
            edge_attr_in = g.edge_attr if mode != "link_only" else None
            z = model(g.x, g.edge_index, edge_attr_in)
            neg = negative_sampling(g.edge_index, g.x.size(0), g.edge_index.size(1)).to(device)

            loss, l, c, s = model.compute_losses(
                z=z,
                edge_attr=g.edge_attr, 
                pos_edge_index=g.edge_index, 
                neg_edge_index=neg,
                pos_weight=pos_weight,
                stance_weight=stance_weight
            )
            total_loss += loss.item()
            total_link += l.item()
            total_conf += c.item()
            total_stance += s.item()

    return (
        total_loss / n_graphs, 
        total_link / n_graphs, 
        total_conf / n_graphs, 
        total_stance / n_graphs
    )

def evaluate_model(model, test_graphs, device, pos_weight, mode: str = "full"):
    """
    Evaluate model on multiple graphs from test set.
    
    Computes comprehensive evaluation metrics:
    - Link AUC: Area under ROC curve for link prediction
    - Confidence MAE: Mean absolute error for confidence prediction
    - Confidence NLL: Negative log-likelihood for confidence prediction
    - Stance NLL: Negative log-likelihood for net (dis)agreement vector prediction
    - Stance KL: KL divergence for net (dis)agreement vector prediction

    Args:
        model (MultitaskDebateGNN): Model to evaluate
        test_graphs (list): List of test graphs
        device (torch.device): Computation device
        pos_weight (torch.Tensor): Positive weight for link prediction
        mode (str, optional): Training mode. Defaults to "full".
    
    Returns:
        dict: Dictionary of evaluation metrics averaged across graphs
    """
    model.eval()
    n = len(test_graphs)
    metrics = {
        "link_auc": 0.0, "conf_mae": 0.0, "conf_nll": 0.0, 
        "stance_nll": 0.0, "stance_kl": 0.0
    }
    
    if n == 0:
        return metrics
        
    with torch.no_grad():
        for g in test_graphs:
            g = g.to(device)
            edge_attr_in = g.edge_attr if mode != "link_only" else None
            z = model(g.x, g.edge_index, edge_attr_in)
        
            # Link AUC
            neg = negative_sampling(g.edge_index, g.x.size(0), g.edge_index.size(1)).to(device)
            pos_logits = model.predict_link(z, g.edge_index)
            neg_logits = model.predict_link(z, neg)
                
            link_pred = torch.cat([pos_logits, neg_logits], dim=0).cpu().numpy()
            link_labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0).cpu().numpy()
            metrics["link_auc"] += float(roc_auc_score(link_labels, link_pred)) / n

            # Edge Attributes
            if mode != "link_only" and hasattr(model, "edge_head"):
                edge_attr_pred = model.predict_edge_attr(z, g.edge_index)
                if edge_attr_pred is None:
                    continue
                    
                # Confidence Metrics
                if mode in ("full", "no_stance"):
                    true_conf = g.edge_attr[:, 0].to(edge_attr_pred.device)
                    conf_mae = F.l1_loss(edge_attr_pred[:, 0], true_conf).item()
                    metrics["conf_mae"] += conf_mae / n
                    
                    conf_probs = edge_attr_pred[:, 0].clamp(min=1e-10, max=1-1e-10)
                    conf_nll = -(true_conf * torch.log(conf_probs) + 
                         (1 - true_conf) * torch.log(1 - conf_probs)).mean().item()
                    metrics["conf_nll"] += conf_nll / n
                
                # Stance Metrics
                if mode in ("full", "no_conf"):
                    true_stance = g.edge_attr[:, 1:].clamp(min=1e-10)
                    stance_probs = edge_attr_pred[:, -3:].clamp(min=1e-10)
                    stance_kl = F.kl_div(torch.log(stance_probs), true_stance, reduction="batchmean").item()
                    metrics["stance_kl"] += stance_kl / n
                    
                    # stance_nll = -torch.sum(true_stance * torch.log(stance_probs)).item()
                    stance_nll = -(true_stance * torch.log(stance_probs)).sum(1).mean().item()
                    metrics["stance_nll"] += stance_nll / n
    
    return {k: v for k, v in metrics.items() if v != 0.0}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
@torch.no_grad()
def get_node_embeddings(model: MultitaskDebateGNN, data: Data,
                        device: str | torch.device | None = None,
                        eval_mode: bool = True) -> torch.Tensor:
    """
    Extract node embeddings from a trained MultitaskDebateGNN model.
    
    Args:
        model (MultitaskDebateGNN): Trained model
        data (Data): PyTorch Geometric data object
        device (str | torch.device | None, optional): Target device. Defaults to None.
        eval_mode (bool, optional): Whether to use evaluation mode. Defaults to True.
    
    Returns:
        torch.Tensor: Node embeddings [N, emb_dim]
    """
    was_training = model.training
    if eval_mode:
        model.eval()    
    if device is not None:
        data = data.to(str(device))
        model = model.to(device)

    edge_attr_in = (data.edge_attr if model.mode != "link_only" else None)
    z = model(data.x, data.edge_index, edge_attr_in)
    
    if eval_mode:
        model.train(was_training)
    return z.detach().cpu()

def roc_auc_score(y_true, y_score):
    """Wrapper for sklearn ROC AUC score computation."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_score)

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducible results across all libraries.
    
    Args:
        seed (int, optional): Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_global_stance_weight(graphs):
    """
    Compute inverse frequency weights for net (dis)agreement classes across all graphs.
    
    Used for class balancing in net (dis)agreement vector prediction loss.
    
    Args:
        graphs (list): List of graph objects with edge_attr containing net (dis)agreement distributions
    
    Returns:
        torch.Tensor: Inverse frequency weights [stance_dim]
    """
    total = torch.zeros(3)
    for g in graphs:
        total += g.edge_attr[:, 1:].sum(0).cpu()
    freq = total.clamp(min=1e-6)
    return freq.sum() / freq

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def _live_plot(live_dict, fold_id):
    """
    Generate live training plots during model training.
    
    Creates two subplots:
    1. Total loss (train vs validation)
    2. Task-specific losses (link, confidence, stance)
    
    Args:
        live_dict (dict): Dictionary containing epoch and loss arrays
        fold_id: Identifier for current training fold
    """
    clear_output(wait=True)
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # Total Loss
    ax = axes[0]
    ax.plot(live_dict["epoch"], live_dict["train_total"], label="Train")
    ax.plot(live_dict["epoch"], live_dict["val_total"],   label="Val")
    ax.set_title(f"Total Loss - Fold {fold_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log Loss")
    ax.set_yscale("log")
    ax.legend()

    # Task Losses
    ax = axes[1]
    for task, color in zip(["link", "conf", "stance"], ["tab:blue", "tab:green", "tab:red"]):
        train_key, val_key = f"train_{task}", f"val_{task}"
        if train_key in live_dict and val_key in live_dict:
            min_len = min(len(live_dict["epoch"]), len(live_dict[train_key]), len(live_dict[val_key]))
            epochs = live_dict["epoch"][:min_len]
            train_loss = live_dict[train_key][:min_len]
            val_loss = live_dict[val_key][:min_len]
            ax.set_yscale("log")
            ax.plot(epochs, train_loss, label=f"{task} train", color=color, linestyle='-')            
            ax.plot(epochs, val_loss, label=f"{task} val", color=color, linestyle='--')

    ax.set_title("Task Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log Loss")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_cv_losses(cv_history, ncols: int = 3, figsize_unit=(4, 3)):
    """
    Plot training/validation loss curves for cross-validation results.
    
    Creates two grid plots:
    1. Total loss per fold
    2. Task-specific losses per fold
    
    Args:
        cv_history (dict): Dictionary mapping fold_id -> training history
        ncols (int, optional): Number of columns in grid. Defaults to 3.
        figsize_unit (tuple, optional): Figure size per subplot. Defaults to (4, 3).
    """
    def make_grid(n_items, title):
        nrows = int(np.ceil(n_items / ncols))
        fig, ax = plt.subplots(
            nrows, ncols, figsize=(figsize_unit[0]*ncols, figsize_unit[1]*nrows),
            constrained_layout=True)
        ax = ax.ravel()
        for extra in range(n_items, len(ax)):
            ax[extra].set_visible(False)
        fig.suptitle(title, fontsize=15)
        return fig, ax

    folds = sorted(cv_history.keys())
    n_folds = len(folds)

    sample_fold = cv_history[folds[0]]
    active_tasks = [
        task for task in ["link", "conf", "stance"]
        if f"train_{task}" in sample_fold and len(sample_fold[f"train_{task}"]) > 0
    ]

    # Total loss plot
    _, ax_total = make_grid(n_folds, "Total Loss per Fold")
    for i, fid in enumerate(folds):
        h = cv_history[fid]
        ax_total[i].plot(h["epoch"], h["train_total"], label="Train", color="tab:blue")
        ax_total[i].plot(h["epoch"], h["val_total"],   label="Val", color="tab:orange")
        ax_total[i].set_title(f"Fold {fid}")
        ax_total[i].set_xlabel("Epoch")
        ax_total[i].set_ylabel("Log Loss")
        ax_total[i].set_yscale("log")
        ax_total[i].legend(fontsize=8)

    # Task-specific plots
    colors = {"link": "tab:blue", "conf": "tab:green", "stance": "tab:red"}
    _, ax_task = make_grid(n_folds, "Task‑specific Loss per Fold")
    for i, fid in enumerate(folds):
        h = cv_history[fid]
        for task in active_tasks:
            tkey = f"train_{task}"
            vkey = f"val_{task}"
    
            min_len = min(len(h["epoch"]), len(h[tkey]), len(h[vkey]))
            epochs = h["epoch"][:min_len]
            train_y = h[tkey][:min_len]
            val_y = h[vkey][:min_len]
            
            # Simple plotting - just plot all data with log scale handling
            train_y_plot = [max(loss, 1e-6) for loss in train_y]  # Handle log scale
            val_y_plot = [max(loss, 1e-6) for loss in val_y]
    
            ax_task[i].plot(epochs, train_y_plot, color=colors[task], ls="-", label=f"{task} train")
            ax_task[i].plot(epochs, val_y_plot, color=colors[task], ls="--", label=f"{task} val")
    
        ax_task[i].set_title(f"Fold {fid}")
        ax_task[i].set_xlabel("Epoch")
        ax_task[i].set_ylabel("Log Loss")
        ax_task[i].set_yscale("log")
        ax_task[i].legend(fontsize=7)

    plt.show()

# Initialize global seed
set_seed(42)