"""
speculative_tree.py
"""

import math
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module

from utils.logits_processor import GreedyProcessor, LogitsProcessor
from .speculative_utils import max_fn, make_tree_stats


# ════════════════════════════════════════════════════════════════════════════
#  数据结构
# ════════════════════════════════════════════════════════════════════════════

class _TreeNode:
    """
    草稿树节点。

    Attributes
    ----------
    token_id  : 本节点对应的 token id
    prob      : 本 token 在父节点处分布中的概率
    depth     : 树深度，从 1 开始
    parent    : 父节点
    children  : 子节点列表
    full_dist : [vocab_size] 概率分布（兄弟节点共享同一张量）
    flat_idx  : BFS 展平列表中的下标
    """

    __slots__ = ["token_id", "prob", "depth", "parent", "children", "full_dist", "flat_idx"]

    def __init__(
        self,
        token_id: int,
        prob: float,
        depth: int,
        parent: Optional["_TreeNode"] = None,
        full_dist: Optional[torch.Tensor] = None,
    ):
        self.token_id  = token_id
        self.prob      = prob
        self.depth     = depth
        self.parent    = parent
        self.children: List["_TreeNode"] = []
        self.full_dist = full_dist
        self.flat_idx: int = -1

    def ancestor_flat_indices(self) -> List[int]:
        """返回从根到本节点（含本节点）的 flat_idx 列表。"""
        indices, node = [], self
        while node is not None and node.flat_idx != -1:
            indices.append(node.flat_idx)
            node = node.parent
        return list(reversed(indices))

    def cum_log_prob(self) -> float:
        lp, node = 0.0, self
        while node is not None:
            lp  += math.log(max(node.prob, 1e-30))
            node = node.parent
        return lp


# ════════════════════════════════════════════════════════════════════════════
#  树注意力掩码（仅供 target 验证阶段使用，drafter 不再用此函数）
# ════════════════════════════════════════════════════════════════════════════

def _build_tree_attention_mask(
    prompt_len: int,
    nodes: List[_TreeNode],
    device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    构造 4-D 加法注意力掩码 (1, 1, prompt_len+N, prompt_len+N)。
    0.0 = 可见；-inf = 屏蔽。
    仅供 target 验证阶段使用，target 须以 attn_implementation="eager" 加载。

    规则：
      - prompt 部分：标准因果掩码
      - 树节点 i：可见全部 prompt + 自身 + 所有祖先节点
    """
    N = len(nodes)
    L = prompt_len + N
    mask = torch.full((L, L), float("-inf"), dtype=dtype, device=device)

    for i in range(prompt_len):
        mask[i, :i + 1] = 0.0

    for node in nodes:
        fi = prompt_len + node.flat_idx
        mask[fi, :prompt_len] = 0.0
        for anc_idx in node.ancestor_flat_indices():
            mask[fi, prompt_len + anc_idx] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, L, L)


# ════════════════════════════════════════════════════════════════════════════
#  树构建：逐层 per-path 批处理，O(max_depth) 次 forward，无需 4-D mask
# ════════════════════════════════════════════════════════════════════════════

def _get_path_tokens(node: "_TreeNode") -> List[int]:
    """返回从根到 node（含 node）的 token id 列表。"""
    path, cur = [], node
    while cur is not None:
        path.append(cur.token_id)
        cur = cur.parent
    return list(reversed(path))


def _build_draft_tree_batched(
    drafter: Module,
    logits_processor: LogitsProcessor,
    base_ids: torch.Tensor,
    start_pos: int,
    max_depth: int,
    branch_factor: int,
    high_thresh: float,
    mid_thresh: float,
    cancel: threading.Event,
    target_device: str,
) -> Tuple[List["_TreeNode"], int]:
    """
    BFS 构建草稿 token 树，每一深度层只做 ONE batched drafter forward。
    forward 次数 = max_depth（含 depth-0 的 prompt forward）
    """
    device     = drafter.device
    prompt_tok = base_ids[0, :start_pos].to(device)

    bfs_nodes:   List[_TreeNode] = []
    total_nodes: int             = 0

    # ── Depth 0：对 prompt 做单次 forward，获取 depth-1 候选分布 ──
    with torch.no_grad():
        out0 = drafter(
            input_ids=prompt_tok.unsqueeze(0),
            past_key_values=None,
            use_cache=False,
        )
    root_probs = logits_processor(
        out0.logits[0, -1:, :]
    ).squeeze(0).to(target_device)

    top_k0 = min(branch_factor, root_probs.shape[-1])
    top_vals0, top_ids0 = root_probs.topk(top_k0)

    current_frontier: List[_TreeNode] = []
    for i in range(top_k0):
        node = _TreeNode(
            token_id=int(top_ids0[i].item()),
            prob=float(top_vals0[i].item()),
            depth=1,
            parent=None,
            full_dist=root_probs,
        )
        node.flat_idx = len(bfs_nodes)
        bfs_nodes.append(node)
        current_frontier.append(node)
        total_nodes += 1

    # ── Depth 1 ~ max_depth-1：逐层展开 ──
    for depth in range(1, max_depth):
        if cancel.is_set() or not current_frontier:
            break

        to_expand = [n for n in current_frontier if n.prob >= mid_thresh]
        if not to_expand:
            break

        seqs = []
        for node in to_expand:
            path_ids = torch.tensor(
                _get_path_tokens(node), dtype=torch.long, device=device
            )
            seq = torch.cat([prompt_tok, path_ids])
            seqs.append(seq)

        batch_input = torch.stack(seqs, dim=0)

        with torch.no_grad():
            out = drafter(
                input_ids=batch_input,
                past_key_values=None,
                use_cache=False,
            )
        last_logits = out.logits[:, -1, :]

        new_frontier: List[_TreeNode] = []
        for idx, node in enumerate(to_expand):
            probs = logits_processor(
                last_logits[idx : idx + 1, :]
            ).squeeze(0).to(target_device)

            node.full_dist = probs

            if node.prob >= high_thresh:
                cur_bf = branch_factor
            else:
                cur_bf = max(1, branch_factor // 2)

            top_k = min(cur_bf, probs.shape[-1])
            top_vals, top_ids = probs.topk(top_k)

            for i in range(top_k):
                child = _TreeNode(
                    token_id=int(top_ids[i].item()),
                    prob=float(top_vals[i].item()),
                    depth=depth + 1,
                    parent=node,
                    full_dist=probs,
                )
                child.flat_idx = len(bfs_nodes)
                bfs_nodes.append(child)
                node.children.append(child)
                new_frontier.append(child)
                total_nodes += 1

        current_frontier = new_frontier

    return bfs_nodes, total_nodes


# ════════════════════════════════════════════════════════════════════════════
#  树拓扑拒绝采样
# ════════════════════════════════════════════════════════════════════════════

def _tree_rejection_sampling(
    nodes: List[_TreeNode],
    target_logits: torch.Tensor,
    prompt_len: int,
    logits_processor: LogitsProcessor,
    skip_sample_adjustment: bool,
    list_tokens_id: List[int],
) -> Tuple[List[int], torch.Tensor]:
    """
    在草稿树上自顶向下做拒绝采样。

    Returns
    -------
    accepted_tokens : 接受的 token id 列表（不含 bonus）
    bonus_token     : 额外追加的 bonus token（标量 tensor）
    """
    def get_children(parent_node: Optional[_TreeNode]) -> List[_TreeNode]:
        if parent_node is None:
            return [n for n in nodes if n.parent is None]
        return parent_node.children

    accepted_tokens: List[int] = []
    current_parent: Optional[_TreeNode] = None

    while True:
        cands = get_children(current_parent)
        if not cands:
            break

        par_logit_pos = (prompt_len - 1) if current_parent is None \
                        else (prompt_len + current_parent.flat_idx)

        p_raw  = target_logits[0, par_logit_pos, :]
        p_dist = logits_processor(p_raw.unsqueeze(0))[0]

        accepted_node: Optional[_TreeNode] = None
        for node in cands:
            p_tok = p_dist[node.token_id].item()
            q_tok = max(node.prob, 1e-9)
            if torch.rand(1, device=target_logits.device).item() <= min(1.0, p_tok / q_tok):
                accepted_node = node
                accepted_tokens.append(node.token_id)
                break

        if accepted_node is None:
            if not skip_sample_adjustment:
                q_sum = torch.zeros_like(p_dist)
                for cand in cands:
                    q_sum[cand.token_id] += cand.prob
                adjusted = torch.clamp(p_dist - q_sum, min=0.0)
                s = adjusted.sum()
                sample_dist = (adjusted / s if s > 1e-9 else p_dist).unsqueeze(0)
            else:
                sample_dist = p_dist.unsqueeze(0)
            return accepted_tokens, logits_processor.sample(sample_dist)

        if accepted_node.token_id in list_tokens_id:
            last_logit_pos = prompt_len + accepted_node.flat_idx
            p_dist_bonus = logits_processor(target_logits[0, last_logit_pos, :].unsqueeze(0))
            return accepted_tokens, logits_processor.sample(p_dist_bonus)

        current_parent = accepted_node

    last_logit_pos = (prompt_len - 1) if current_parent is None \
                     else (prompt_len + current_parent.flat_idx)
    p_dist_bonus = logits_processor(target_logits[0, last_logit_pos, :].unsqueeze(0))
    return accepted_tokens, logits_processor.sample(p_dist_bonus)


# ════════════════════════════════════════════════════════════════════════════
#  后台预生成线程
# ════════════════════════════════════════════════════════════════════════════

def _tree_pregen_thread_fn(
    drafter: Module,
    logits_processor: LogitsProcessor,
    base_ids: torch.Tensor,
    start_pos: int,
    next_cg: int,
    branch_factor: int,
    high_thresh: float,
    mid_thresh: float,
    vocabulary_size: int,
    target_device: str,
    out: dict,
    cancel: threading.Event,
    debug: bool = False,
) -> None:
    try:
        bfs_nodes, tree_nodes = _build_draft_tree_batched(
            drafter, logits_processor, base_ids, start_pos, next_cg,
            branch_factor, high_thresh, mid_thresh, cancel, target_device,
        )
        out["tree_nodes"] = tree_nodes
        out["num_leaves"] = sum(1 for n in bfs_nodes if not n.children)

        if cancel.is_set() or not bfs_nodes:
            out["valid"] = False
            return

        out.update({
            "valid":     True,
            "nodes":     bfs_nodes,
            "base_ids":  base_ids,
            "start_pos": start_pos,
        })

        if debug:
            print(f"[TREE-PREGEN] ✅ nodes={tree_nodes}, "
                  f"leaves={out['num_leaves']}, gamma={next_cg}")

    except Exception as exc:
        if debug:
            import traceback
            print(f"[TREE-PREGEN] ❌ Error: {exc}")
            traceback.print_exc()
        out["valid"] = False


# ════════════════════════════════════════════════════════════════════════════
#  主函数
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def speculative_generate_tree_pregen(
    inputs: List[int],
    drafter: Module,
    target: Module,
    tokenizer=None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id=1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
    debug: bool = False,
    branch_factor: int = 3,
    high_thresh: float = 0.7,
    mid_thresh: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    树形预生成投机解码
    Returns: (output_ids, accept_rate, tree_stats_dict)
    """
    # ── 可选种子设置 ──────────────────────────────────────────
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    # ──────────────────────────────────────────────────────────

    use_cache = False

    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    list_tokens_id = [t for t in list_tokens_id if t is not None]
    stop_tokens = (torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
                if list_tokens_id else
                torch.tensor([], dtype=torch.long, device=target.device).unsqueeze(1))

    drafts_accepted, drafts_speculated = 0.0, 0.0
    tree_attempts = tree_hits = tree_total_nodes = tree_total_leaves = 0

    vocabulary_size = target.config.vocab_size
    prompt_len      = len(inputs)
    max_seq_length  = (
        target.config.max_position_embeddings
        if hasattr(target.config, "max_position_embeddings")
        else (target.config.max_context_length
              if hasattr(target.config, "max_context_length") else 1024)
    )
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full(
        (1, total_len), pad_token_id, dtype=torch.long, device=target.device
    )
    input_ids[0, :prompt_len] = torch.tensor(
        inputs, dtype=torch.long, device=target.device
    )
    current_position = prompt_len

    # ── 初始 target 预热 ──
    if first_target:
        Mp0 = target(
            input_ids=input_ids[..., :current_position],
            past_key_values=None,
            use_cache=False,
        )
        p_p = logits_processor(Mp0.logits[..., -1, :])
        t   = logits_processor.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1
        if stop_tokens.numel() > 0 and torch.isin(t, stop_tokens):
            return (
                input_ids[0, prompt_len:current_position].tolist(),
                0.0,
                make_tree_stats(tree_attempts, tree_hits, tree_total_nodes, tree_total_leaves),
            )

    prev_tree_thread: Optional[threading.Thread] = None
    prev_tree_out:    Optional[dict]             = None
    prev_cancel:      Optional[threading.Event]  = None
    prev_next_cg:     int                        = 0

    # ══════════════════ 主循环 ══════════════════
    while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        if corrected_gamma <= 0:
            break

        # ── Phase 1: 等待上一轮后台线程 ──
        use_tree = False
        if prev_tree_thread is not None:
            prev_tree_thread.join()
            prev_tree_thread = None
            tree_attempts += 1

            valid = (
                prev_tree_out is not None
                and prev_tree_out.get("valid", False)
                and prev_tree_out.get("start_pos") == current_position
                and prev_next_cg == corrected_gamma
            )
            if valid:
                tree_hits         += 1
                tree_total_nodes  += prev_tree_out["tree_nodes"]
                tree_total_leaves += prev_tree_out["num_leaves"]
                use_tree = True
                if debug:
                    print(f"[TREE] ✅ Hit! nodes={prev_tree_out['tree_nodes']}, "
                          f"leaves={prev_tree_out['num_leaves']}")
            else:
                if prev_tree_out:
                    tree_total_nodes  += prev_tree_out.get("tree_nodes", 0)
                    tree_total_leaves += prev_tree_out.get("num_leaves", 0)
                if debug:
                    reason = (
                        "pos mismatch"
                        if prev_tree_out and prev_tree_out.get("start_pos") != current_position
                        else "gamma mismatch or build failed"
                    )
                    print(f"[TREE] ❌ Miss ({reason})")

        n_accepted: int      = 0
        effective_gamma: int = corrected_gamma

        if use_tree:
            # ── 树形并行验证路径 ──
            nodes    = prev_tree_out["nodes"]
            base_ids = prev_tree_out["base_ids"].to(target.device)

            tree_tok   = torch.tensor(
                [n.token_id for n in nodes], dtype=torch.long, device=target.device
            ).unsqueeze(0)
            flat_input = torch.cat([base_ids[..., :current_position], tree_tok], dim=1)
            attn_mask  = _build_tree_attention_mask(current_position, nodes, target.device)

            # effective_gamma = 树的最大深度，与线性 gamma 量纲一致
            effective_gamma   = max((n.depth for n in nodes), default=corrected_gamma)

            # ── v4.2 修正：用树的最大深度作为分母，与线性方法对齐 ──
            drafts_speculated += effective_gamma

            Mp = target(
                input_ids      = flat_input,
                attention_mask = attn_mask,
                past_key_values= None,
                use_cache      = False,
            )

            accepted_tokens, bonus_token = _tree_rejection_sampling(
                nodes                  = nodes,
                target_logits          = Mp.logits,
                prompt_len             = current_position,
                logits_processor       = logits_processor,
                skip_sample_adjustment = skip_sample_adjustment,
                list_tokens_id         = list_tokens_id,
            )

            n_accepted      = len(accepted_tokens)
            drafts_accepted += n_accepted

            for pos, tok in enumerate(accepted_tokens):
                input_ids[0, current_position + pos] = tok
            input_ids[0, current_position + n_accepted] = bonus_token.item()

            clear_end = current_position + effective_gamma
            if current_position + n_accepted + 1 < clear_end:
                input_ids[0, current_position + n_accepted + 1 : clear_end] = pad_token_id

            current_position += n_accepted + 1

            if debug:
                try:
                    gen = tokenizer.decode(
                        input_ids[0, prompt_len:current_position].tolist(),
                        skip_special_tokens=True,
                    )
                    print(f"[TREE] accepted={n_accepted}/{effective_gamma} | ...{gen[-60:]}")
                except Exception:
                    pass

            if stop_tokens.numel() > 0 and torch.isin(bonus_token, stop_tokens):
                break

            stop_in_accepted = next(
                (pos for pos, tok in enumerate(accepted_tokens)
                 if tok in list_tokens_id),
                None,
            )
            if stop_in_accepted is not None:
                if prev_tree_thread is not None:
                    prev_cancel.set()
                    prev_tree_thread.join()
                return (
                    input_ids[0, prompt_len:current_position].tolist(),
                    drafts_accepted / max(drafts_speculated, 1),
                    make_tree_stats(tree_attempts, tree_hits, tree_total_nodes, tree_total_leaves),
                )

        else:
            # ── 线性后备路径（标准投机解码）──
            q = torch.zeros(
                (1, effective_gamma, vocabulary_size), device=target.device
            )
            input_ids = input_ids.to(drafter.device)
            for k in range(effective_gamma):
                Mq = drafter(
                    input_ids=input_ids[..., :current_position + k],
                    past_key_values=None,
                    use_cache=False,
                )
                probs = logits_processor(Mq.logits[..., -1, :])
                q[0, k] = probs.to(target.device)
                input_ids[0, current_position + k] = logits_processor.sample(probs)
            input_ids = input_ids.to(target.device)

            drafts_speculated += effective_gamma

            Mp = target(
                input_ids=input_ids[..., :current_position + effective_gamma],
                past_key_values=None,
                use_cache=False,
            )
            p = logits_processor(
                Mp.logits[
                    ...,
                    current_position - 1 : current_position + effective_gamma - 1,
                    :,
                ]
            )

            r = torch.rand(effective_gamma, device=target.device)
            fractions = p / q
            n = effective_gamma
            for i in range(effective_gamma):
                if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                    n = i
                    break

            n_accepted      = n
            drafts_accepted += n

            stop_locs = torch.nonzero(
                torch.eq(
                    input_ids[..., current_position : current_position + n],
                    stop_tokens,
                )
            )
            if stop_locs.shape[0] > 0:
                stop_loc = stop_locs[0, 1].item()
                if prev_tree_thread is not None:
                    prev_cancel.set()
                    prev_tree_thread.join()
                return (
                    input_ids[0, prompt_len : current_position + stop_loc + 1].tolist(),
                    drafts_accepted / max(drafts_speculated, 1),
                    make_tree_stats(tree_attempts, tree_hits, tree_total_nodes, tree_total_leaves),
                )

            if n == effective_gamma:
                p_p = logits_processor(
                    Mp.logits[..., current_position + effective_gamma - 1, :]
                )
            else:
                p_p = (
                    max_fn(p[..., n, :] - q[0, n, :])
                    if not skip_sample_adjustment
                    else p[..., n, :]
                )
            actual_bonus = logits_processor.sample(p_p)

            input_ids[0, current_position + n : current_position + effective_gamma] = pad_token_id
            input_ids[0, current_position + n] = actual_bonus
            current_position += n + 1

            if debug:
                try:
                    gen = tokenizer.decode(
                        input_ids[0, prompt_len:current_position].tolist(),
                        skip_special_tokens=True,
                    )
                    print(f"[FALLBACK] accepted={n}/{effective_gamma} | ...{gen[-60:]}")
                except Exception:
                    pass

            if stop_tokens.numel() > 0 and torch.isin(actual_bonus, stop_tokens):
                break

        # ── Phase 7: 启动下一轮树预生成后台线程（无条件启动）──
        next_cg = min(gamma, total_len - current_position - 1)
        if next_cg > 0:
            cancel_event = threading.Event()
            tree_out     = {}
            _ids_snap    = input_ids.clone()
            _cur_pos     = current_position
            _ng          = next_cg

            t_tree = threading.Thread(
                target = _tree_pregen_thread_fn,
                kwargs = dict(
                    drafter          = drafter,
                    logits_processor = logits_processor,
                    base_ids         = _ids_snap,
                    start_pos        = _cur_pos,
                    next_cg          = _ng,
                    branch_factor    = branch_factor,
                    high_thresh      = high_thresh,
                    mid_thresh       = mid_thresh,
                    vocabulary_size  = vocabulary_size,
                    target_device    = str(target.device),
                    out              = tree_out,
                    cancel           = cancel_event,
                    debug            = debug,
                ),
                name   = "TreePregen",
                daemon = True,
            )
            t_tree.start()

            prev_tree_thread = t_tree
            prev_tree_out    = tree_out
            prev_cancel      = cancel_event
            prev_next_cg     = next_cg
        else:
            prev_tree_thread = prev_tree_out = prev_cancel = None
            prev_next_cg = 0

    # ── 清理后台线程 ──
    if prev_tree_thread is not None:
        if prev_cancel is not None:
            prev_cancel.set()
        prev_tree_thread.join()

    return (
        input_ids[0, prompt_len : min(current_position, total_len)].tolist(),
        drafts_accepted / max(drafts_speculated, 1),
        make_tree_stats(tree_attempts, tree_hits, tree_total_nodes, tree_total_leaves),
    )