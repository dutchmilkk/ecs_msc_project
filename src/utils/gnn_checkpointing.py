"""
Checkpoint saving/loading utilities.

Usage:
- Save:
    from src.utils.gnn_checkpointing import save_model_checkpoint
    path = save_model_checkpoint(model, model_args, train_args, out_dir="checkpoints", prefix="best_model")

- Load:
    from src.utils.gnn_checkpointing import load_model_checkpoint
    model, ckpt = load_model_checkpoint(path, device=device)
    # If use a different model class:
    # model, ckpt = load_model_checkpoint(path, device=device, model_class_path="your.module.Model")

Notes:
- Converts class objects in model_args (conv_cls, conv_cls_list) to import paths for safe pickling.
- Reconstructs those classes on load. Default model class is src.models.multitask_debate_gnn.MultitaskDebateGNN.
"""

import os
import importlib
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch


def _qualname(obj) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _resolve(path: str):
    module_path, _, attr_path = path.rpartition(".")
    mod = importlib.import_module(module_path)
    obj = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def serialize_model_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make model_args pickle-safe by turning class objects into import paths.
    Handles keys: 'conv_cls' and 'conv_cls_list'.
    """
    safe = dict(args)
    if "conv_cls" in safe and hasattr(safe["conv_cls"], "__qualname__"):
        safe["conv_cls"] = _qualname(safe["conv_cls"])
    if "conv_cls_list" in safe and isinstance(safe["conv_cls_list"], list):
        safe["conv_cls_list"] = [
            _qualname(c) if hasattr(c, "__qualname__") else c
            for c in safe["conv_cls_list"]
        ]
    return safe


def save_model_checkpoint(
    model: torch.nn.Module,
    model_args: Dict[str, Any],
    train_args: Dict[str, Any],
    out_dir: str = "checkpoints",
    prefix: str = "model",
    timestamp: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%y%m%d%H%M")
    out_path = os.path.join(out_dir, f"{prefix}_{ts}.pth")
    
    checkpoint: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model_args": serialize_model_args(model_args),
        "train_args": train_args,
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, out_path)
    print(f"Saved model checkpoint to {out_path}")
    return out_path


def load_model_checkpoint(
    path: str,
    device: str | torch.device = "cpu",
    model_class_path: str = "src.models.multitask_debate_gnn.MultitaskDebateGNN",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    print(f"Loading model checkpoint from {path}")
    ckpt: Dict[str, Any] = torch.load(path, map_location=device)

    margs = dict(ckpt["model_args"])
    # Resolve conv classes from strings
    if isinstance(margs.get("conv_cls"), str):
        margs["conv_cls"] = _resolve(margs["conv_cls"])
    if isinstance(margs.get("conv_cls_list"), list):
        margs["conv_cls_list"] = [
            _resolve(x) if isinstance(x, str) else x for x in margs["conv_cls_list"]
        ]

    model_cls = _resolve(model_class_path)
    model = model_cls(**margs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt