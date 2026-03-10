"""
speculative_pregeneration.py
"""

import threading
from typing import List, Tuple

import torch
from torch.nn import Module

from utils.logits_processor import GreedyProcessor, LogitsProcessor
from .speculative_utils import max_fn


@torch.no_grad()
def speculative_generate_pregeneration(
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
) -> Tuple[List[int], float]:
    """
    串行预生成投机解码（修复版 v1.2）。
    Returns: (output_ids, accept_rate)
    """
    use_cache = False

    list_tokens_id = eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    list_tokens_id = [t for t in list_tokens_id if t is not None]
    stop_tokens = (torch.tensor(list_tokens_id, dtype=torch.long, device=target.device).unsqueeze(1)
                if list_tokens_id else
                torch.tensor([], dtype=torch.long, device=target.device).unsqueeze(1))

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
        Mp0 = target(input_ids=input_ids[..., :current_position], past_key_values=None, use_cache=False)
        p_p = logits_processor(Mp0.logits[..., -1, :])
        first_tok = logits_processor.sample(p_p)
        input_ids[0, current_position] = first_tok
        current_position += 1
        if stop_tokens.numel() > 0 and torch.isin(first_tok, stop_tokens):
            return input_ids[0, prompt_len:current_position].tolist(), 0.0

    prev_thread  = None
    prev_out     = None
    prev_cancel  = None
    prev_next_cg = 0

    while current_position < total_len:
        corrected_gamma = min(gamma, total_len - current_position - 1)
        if corrected_gamma <= 0:
            break

        # ── Phase 1: join 上一轮后台线程，校验 start_pos 和 gamma ──
        use_pregenerated = False
        if prev_thread is not None:
            prev_thread.join()
            prev_thread = None
            if (prev_out is not None
                    and prev_out.get("valid", False)
                    and prev_out.get("start_pos") == current_position   # 位置必须对齐
                    and prev_next_cg == corrected_gamma):               # gamma 必须匹配
                use_pregenerated = True

        # ── Phase 2: 草稿生成 ──
        q = torch.zeros((1, corrected_gamma, vocabulary_size), device=target.device)

        if use_pregenerated:
            q         = prev_out["q"]
            input_ids = prev_out["input_ids"].to(target.device)
            if debug:
                print(f"[SERIAL-PREGEN] ✅ 使用预生成草稿，pos={current_position}")
        else:
            if debug and prev_out is not None:
                print(f"[SERIAL-PREGEN] ❌ 预生成未命中，重新生成草稿")
            input_ids = input_ids.to(drafter.device)
            for k in range(corrected_gamma):
                Mq = drafter(
                    input_ids=input_ids[..., :current_position + k],
                    past_key_values=None, use_cache=False,
                )
                probs = logits_processor(Mq.logits[..., -1, :])
                q[0, k] = probs.to(target.device)
                input_ids[0, current_position + k] = logits_processor.sample(probs)
            input_ids = input_ids.to(target.device)

        drafts_speculated += corrected_gamma

        # ── Phase 3: target 验证 ──
        Mp = target(
            input_ids=input_ids[..., :current_position + corrected_gamma],
            past_key_values=None, use_cache=False,
        )
        p = logits_processor(
            Mp.logits[..., current_position - 1:current_position + corrected_gamma - 1, :]
        )

        # ── Phase 4: 拒绝采样 ──
        r         = torch.rand(corrected_gamma, device=target.device)
        fractions = p / q
        n = corrected_gamma
        for i in range(corrected_gamma):
            if r[i] > fractions[0, i, input_ids[0, current_position + i]]:
                n = i
                break

        drafts_accepted += n

        stop_locs = torch.nonzero(
            torch.eq(input_ids[..., current_position:current_position + n], stop_tokens)
        )
        if stop_locs.shape[0] > 0:
            stop_loc = stop_locs[0, 1].item()
            if prev_thread is not None:
                prev_cancel.set()
                prev_thread.join()
            return (
                input_ids[0, prompt_len:current_position + stop_loc + 1].tolist(),
                drafts_accepted / max(drafts_speculated, 1),
            )

        # ── Phase 5: bonus token ──
        if n == corrected_gamma:
            p_p = logits_processor(Mp.logits[..., current_position + corrected_gamma - 1, :])
        else:
            p_p = max_fn(p[..., n, :] - q[0, n, :]) if not skip_sample_adjustment else p[..., n, :]
        actual_bonus = logits_processor.sample(p_p)

        # ── Phase 6: 更新序列 ──
        input_ids[0, current_position + n:current_position + corrected_gamma] = pad_token_id
        input_ids[0, current_position + n] = actual_bonus
        current_position += n + 1

        if debug:
            try:
                print(f"[SERIAL-PREGEN] accepted={n}/{corrected_gamma} | "
                      f"{tokenizer.decode(input_ids[0, prompt_len:current_position].tolist(), skip_special_tokens=True)[-60:]}")
            except Exception:
                pass

        # ── v1.2 修正：用 actual_bonus 判断停止，不再用 first_tok ──
        if stop_tokens.numel() > 0 and torch.isin(actual_bonus, stop_tokens):
            break

        next_cg = min(gamma, total_len - current_position - 1)

        # ── Phase 7: 无条件启动后台预生成线程（v1.2：不再要求全部接受）──
        if next_cg > 0:
            cancel_event = threading.Event()
            pregen_out   = {}
            _ids_snap = input_ids.clone()
            _cur_pos  = current_position   # 已经更新过的新位置
            _ng       = next_cg

            def _serial_pregen(
                ids=_ids_snap, pos=_cur_pos, ng=_ng,
                out=pregen_out, cancel=cancel_event,
            ):
                with torch.no_grad():
                    pg_ids = ids.clone().to(drafter.device)
                    pg_q   = torch.zeros((1, ng, vocabulary_size), device=target.device)
                    for k in range(ng):
                        if cancel.is_set():
                            out["valid"] = False
                            return
                        Mq    = drafter(input_ids=pg_ids[..., :pos + k], past_key_values=None, use_cache=False)
                        probs = logits_processor(Mq.logits[..., -1, :])
                        pg_q[0, k] = probs.to(target.device)
                        pg_ids[0, pos + k] = logits_processor.sample(probs)
                    out["valid"]     = True
                    out["q"]         = pg_q
                    out["input_ids"] = pg_ids.to(target.device)
                    out["start_pos"] = pos   # 记录位置，供主线程校验

            _pregen_thread = threading.Thread(target=_serial_pregen, daemon=True, name="SerialPregen")
            _pregen_thread.start()

            prev_thread  = _pregen_thread
            prev_out     = pregen_out
            prev_cancel  = cancel_event
            prev_next_cg = next_cg
        else:
            prev_thread = prev_out = prev_cancel = None
            prev_next_cg = 0

    if prev_thread is not None:
        if prev_cancel is not None:
            prev_cancel.set()
        prev_thread.join()

    return (
        input_ids[0, prompt_len:min(current_position, total_len)].tolist(),
        drafts_accepted / max(drafts_speculated, 1),
    )