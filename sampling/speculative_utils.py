"""
speculative_utils.py
共享工具函数，供其他投机解码模块使用。
"""

from typing import Any, Dict

import torch


def max_fn(x: torch.Tensor) -> torch.Tensor:
    """拒绝采样后修正概率分布：norm(max(0, x))。"""
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum


def make_tree_stats(
    attempts: int,
    hits: int,
    total_nodes: int,
    total_leaves: int,
) -> Dict[str, Any]:
    denom = max(attempts, 1)
    return {
        "tree_attempts"      : attempts,
        "tree_hits"          : hits,
        "hit_rate"           : hits / denom,
        "total_nodes"        : total_nodes,
        "avg_nodes_per_tree" : total_nodes  / denom,
        "total_leaves"       : total_leaves,
        "avg_leaves_per_tree": total_leaves / denom,
    }