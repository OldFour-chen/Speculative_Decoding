"""
speculative_standard.py
标准投机解码（Leviathan et al., 2023）。
"""

from typing import List, Tuple

import torch
from torch.nn import Module

from utils.logits_processor import GreedyProcessor, LogitsProcessor
from utils.caching import prune_cache
import utils.printing as printing

from .speculative_utils import max_fn


@torch.no_grad()
def speculative_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    tokenizer=None,
    gamma: int = 5,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    max_gen_len: int = 40,
    eos_tokens_id=1,
    pad_token_id: int = 0,
    use_cache: bool = True,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
    debug: bool = False,
) -> Tuple[List[int], float]:
    """
    标准投机解码。
    Returns: (output_ids, accept_rate)
    """
    drafter_cache, target_cache = None, None

    # 兼容 Qwen：eos_tokens_id 可能是 None 或包含 None 的列表
    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    list_tokens_id = [t for t in list_tokens_id if t is not None]  # 过滤 None

    if len(list_tokens_id) == 0:
        # 没有有效的 eos token，构造空张量，isin 检查永远不会触发
        stop_tokens = torch.tensor([], dtype=torch.long, device=target.device).unsqueeze(1)
    else:
        stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)

    drafts_accepted, drafts_speculated = 0.0, 0.0
    vocabulary_size = target.config.vocab_size

    prompt_len = len(inputs)
    max_seq_length = (
        target.config.max_position_embeddings
        if hasattr(target.config, "max_position_embeddings")
        else (target.config.max_context_length if hasattr(target.config, "max_context_length") else 1024)
    )
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=target.device)
    current_position = prompt_len

    if first_target:
        Mp = target(
            input_ids=input_ids[..., :current_position],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        p_p = logits_processor(Mp.logits[..., -1, :])
        t = logits_processor.sample(p_p)
        input_ids[0, current_position] = t
        current_position += 1
        if stop_tokens.numel() > 0 and torch.isin(t, stop_tokens):
            if debug:
                printing.end_token_found(0)
            return input_ids[0, prompt_len:current_position].tolist(), 0.0

    while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)
        input_ids = input_ids.to(drafter.device)

        for k in range(corrected_gamma):
            Mq = drafter(
                input_ids=input_ids[..., :current_position + k],
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Mq.past_key_values
            draft_probs = logits_processor(Mq.logits[..., -1, :])
            q[0, k] = draft_probs.to(target.device)
            xi = logits_processor.sample(draft_probs)
            input_ids[0, current_position + k] = xi

        drafts_speculated += corrected_gamma
        input_ids = input_ids.to(target.device)

        Mp = target(
            input_ids=input_ids[..., :current_position + corrected_gamma],
            past_key_values=target_cache,
            use_cache=use_cache,
        )
        target_cache = Mp.past_key_values
        p = logits_processor(Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :])

        r = torch.rand(corrected_gamma, device=target.device)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                n = i
                break

        drafts_accepted += n

        if stop_tokens.numel() > 0:
            stop_locs = torch.nonzero(
                torch.eq(input_ids[..., current_position:current_position + n], stop_tokens)
            )
            if stop_locs.shape[0] > 0:
                stop_loc = stop_locs[0, 1].item()
                return (
                    input_ids[0, prompt_len:current_position + stop_loc + 1].tolist(),
                    drafts_accepted / max(drafts_speculated, 1),
                )

        if n == corrected_gamma:
            p_p = logits_processor(Mp.logits[..., current_position + corrected_gamma - 1, :])
        else:
            if use_cache:
                drafter_cache = prune_cache(drafter_cache, corrected_gamma - n)
                target_cache  = prune_cache(target_cache,  corrected_gamma - n + 1)
            p_p = max_fn(p[..., n, :] - q[0, n, :]) if not skip_sample_adjustment else p[..., n, :]
        x = logits_processor.sample(p_p)

        if debug:
            generated = input_ids.clone().detach()

        input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
        input_ids[0, current_position + n] = x

        if debug:
            printing.speculative_step(tokenizer, generated, input_ids, n, prompt_len,
                                      current_position, corrected_gamma)

        current_position += n + 1

        if stop_tokens.numel() > 0 and torch.isin(x, stop_tokens):
            return input_ids[0, prompt_len:current_position].tolist(), drafts_accepted / max(drafts_speculated, 1)

    return input_ids[0, prompt_len:].tolist(), drafts_accepted / max(drafts_speculated, 1)