import argparse
import random
import numpy as np
import torch
from sampling import autoregressive_generate
from sampling.speculative_decoding import (
    speculative_generate,
    speculative_generate_pregeneration,
    speculative_generate_tree_pregen,
)
from utils.logits_processor import (
    GreedyProcessor, MultinomialProcessor, TopKProcessor,
    NucleusProcessor, TopKNucleusProcessor,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
)
import time
import os
from termcolor import colored
from power import PowerMonitor


# ─────────────────────────── 测试场景 ───────────────────────────
test_scenarios = [
    {
        "name": "Route Planning",
        "prompt": (
            "As an unmanned aerial vehicle (UAV) flight control system, please plan an optimal "
            "flight route from the starting point A (120.5°E, 30.2°N) to the destination B "
            "(120.8°E, 30.5°N), avoiding no-fly zones. Please provide detailed waypoint "
            "coordinates and flight parameters."
        ),
    },
    {
        "name": "Obstacle Detection",
        "prompt": (
            "An obstacle has been detected ahead of the UAV, at a distance of 50 meters, "
            "with a height of 15 meters and a width of 10 meters. The current flight speed "
            "is 12m/s and the altitude is 80 meters. Please immediately provide an obstacle "
            "avoidance strategy, including maneuvering actions and parameter adjustments."
        ),
    },
    {
        "name": "Battery Management",
        "prompt": (
            "The current battery level is 30%, the distance to the return-to-home (RTH) point "
            "is 5 kilometers, the wind speed is 3m/s (headwind), the current altitude is 120 "
            "meters, and the payload is 2kg. Please evaluate whether an immediate return to "
            "home is necessary and provide a return route plan."
        ),
    },
]


class InferenceCLI:

    TARGET_DEVICE  = "cuda:0"
    DRAFTER_DEVICE = "cuda:2"

    def __init__(self, target_device: str = TARGET_DEVICE, drafter_device: str = DRAFTER_DEVICE):
        self.target_device  = target_device
        self.drafter_device = drafter_device

        # ── 超参 ──
        self.gamma    = 4
        self.gen_len  = 150
        self.debug    = False

        # ── 树形预生成参数 ──
        self.branch_factor = 3
        self.high_thresh   = 0.7
        self.mid_thresh    = 0.3

        # ── 运行开关 ──
        self.run_spec        = True
        self.run_spec_pregen = True
        self.run_spec_tree   = True
        self.run_target_ar   = True
        self.run_drafter_ar  = True

        self.cache      = False
        self.target_gen = True
        self.chat       = True

        self.processor = GreedyProcessor()

        self._load_models()
        self._run()

    # ─────────────────────────── 模型加载 ───────────────────────────
    def _load_models(self):
        modelscope_cache = os.path.expanduser("~/.cache/modelscope/hub/models/")

        target_model = os.getenv(
            "TARGET_MODEL_PATH",
            os.path.join(modelscope_cache, "LLM-Research/Meta-Llama-3-8B-Instruct"),
        )
        drafter_model = os.getenv(
            "DRAFTER_MODEL_PATH",
            os.path.join(modelscope_cache, "LLM-Research/Llama-3.2-1B-Instruct"),
        )

        target_quantize  = QuantoConfig(weights="int8")
        drafter_quantize = QuantoConfig(weights="int8")

        dual_gpu = self.target_device != self.drafter_device

        for path, name, cmd in [
            (target_model,  "Target",  "modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct"),
            (drafter_model, "Drafter", "modelscope download --model LLM-Research/Llama-3.2-1B-Instruct"),
        ]:
            if not os.path.exists(path):
                print(colored(f"Warning: {name} model path not found: {path}", "red"))
                print(colored(f"  → {cmd}", "yellow"))

        print(colored("Loading models...", "light_grey"))

        self.target = AutoModelForCausalLM.from_pretrained(
            target_model,
            quantization_config=target_quantize,
            device_map=self.target_device,
            trust_remote_code=True,
        )
        self.target.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)

        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_model,
            attn_implementation="eager",
            quantization_config=drafter_quantize,
            device_map=self.drafter_device,
            trust_remote_code=True,
        )
        self.drafter.eval()

        self.end_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        def _gpu_idx(dev: str) -> int:
            if dev in ("cuda", "cpu"):
                return 0
            return int(dev.split(":")[-1])

        target_gpu_idx  = _gpu_idx(self.target_device)
        drafter_gpu_idx = _gpu_idx(self.drafter_device)

        if dual_gpu:
            self.power_monitor         = PowerMonitor(gpu_index=target_gpu_idx,  poll_interval=0.001)
            self.power_monitor_drafter = PowerMonitor(gpu_index=drafter_gpu_idx, poll_interval=0.001)
            self._dual_gpu = True
        else:
            self.power_monitor         = PowerMonitor(gpu_index=target_gpu_idx, poll_interval=0.001)
            self.power_monitor_drafter = None
            self._dual_gpu = False

    # ─────────────────────────── 能耗辅助 ───────────────────────────
    def _pm_start(self):
        self.power_monitor.start()
        if self._dual_gpu and self.power_monitor_drafter:
            self.power_monitor_drafter.start()

    def _pm_stop(self):
        self.power_monitor.stop()
        if self._dual_gpu and self.power_monitor_drafter:
            self.power_monitor_drafter.stop()

    def _pm_energy_separate(self) -> tuple[float, float]:
        """返回 (target_energy_J, drafter_energy_J)，单卡时 drafter = 0.0。"""
        target_energy  = self.power_monitor.calculate_energy()
        drafter_energy = (
            self.power_monitor_drafter.calculate_energy()
            if self._dual_gpu and self.power_monitor_drafter
            else 0.0
        )
        return target_energy, drafter_energy

    def _sync_all(self):
        for dev in {self.target_device, self.drafter_device}:
            if dev.startswith("cuda"):
                torch.cuda.synchronize(dev)

    # ─────────────────────────── 单场景推理 ───────────────────────────
    def _infer(self, scenario_name: str, prefix: str):
        sep = "=" * 80
        print(colored(f"\n{sep}", "white", attrs=["bold"]))
        print(colored(f"场景: {scenario_name}", "white", attrs=["bold"]))
        print(colored(f"Prompt: {prefix}", "white"))
        print(colored(sep, "white", attrs=["bold"]))

        if self.chat:
            prefix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prefix}],
                add_generation_prompt=True,
                tokenize=False,
            )

        tokenized = self.tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()

        def _empty_result(**extra):
            base = {
                "output": None, "throughput": 0.0,
                "target_energy_j": 0.0, "drafter_energy_j": 0.0,
                "elapsed_s": 0.0,       # ← 新增
                "enabled": False,
            }
            base.update(extra)
            return base

        results = {
            "spec":        _empty_result(accept_rate=0.0),
            "spec_pregen": _empty_result(accept_rate=0.0),
            "spec_tree":   _empty_result(accept_rate=0.0, tree_stats={}),
            "target":      _empty_result(),
            "drafter":     _empty_result(),
        }

        # ══════════════════ 1. 标准投机解码 ══════════════════
        if self.run_spec:
            results["spec"]["enabled"] = True
            self._set_seed(42)
            self._pm_start()
            t0 = time.time()

            output_ids, accept_rate = speculative_generate(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=self.cache,
            )

            self._sync_all()
            t1 = time.time()
            self._pm_stop()

            te, de   = self._pm_energy_separate()
            out_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            results["spec"].update({
                "output": out_text, "accept_rate": accept_rate,
                "throughput": len(out_text) / (t1 - t0),
                "target_energy_j": te, "drafter_energy_j": de,
                "elapsed_s": t1 - t0,  # ← 新增
            })

            _print_block(
                "Speculative (标准)", out_text, results["spec"],
                color="green",
                extra_lines=[f"Accept Rate : {accept_rate:.3f}"],
                dual_gpu=self._dual_gpu,
            )

        # ══════════════════ 2. 串行预生成投机解码 ══════════════════
        if self.run_spec_pregen:
            results["spec_pregen"]["enabled"] = True
            self._set_seed(42)
            self._pm_start()
            t0 = time.time()

            output_ids, accept_rate = speculative_generate_pregeneration(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=False,
            )

            self._sync_all()
            t1 = time.time()
            self._pm_stop()

            te, de   = self._pm_energy_separate()
            out_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            results["spec_pregen"].update({
                "output": out_text, "accept_rate": accept_rate,
                "throughput": len(out_text) / (t1 - t0),
                "target_energy_j": te, "drafter_energy_j": de,
                "elapsed_s": t1 - t0,  # ← 新增
            })

            _print_block(
                "Speculative + Serial Pre-Gen", out_text, results["spec_pregen"],
                color="yellow",
                extra_lines=[f"Accept Rate : {accept_rate:.3f}"],
                dual_gpu=self._dual_gpu,
            )

        # ══════════════════ 3. 树形预生成投机解码 ══════════════════
        if self.run_spec_tree:
            results["spec_tree"]["enabled"] = True
            self._set_seed(42)
            self._pm_start()
            t0 = time.time()

            output_ids, accept_rate, tree_stats = speculative_generate_tree_pregen(
                tokenized,
                self.drafter,
                self.target,
                tokenizer=self.tokenizer,
                logits_processor=self.processor,
                gamma=self.gamma,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                debug=self.debug,
                use_cache=False,
                branch_factor=self.branch_factor,
                high_thresh=self.high_thresh,
                mid_thresh=self.mid_thresh,
            )

            self._sync_all()
            t1 = time.time()
            self._pm_stop()

            te, de   = self._pm_energy_separate()
            out_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            results["spec_tree"].update({
                "output": out_text, "accept_rate": accept_rate,
                "throughput": len(out_text) / (t1 - t0),
                "target_energy_j": te, "drafter_energy_j": de,
                "elapsed_s": t1 - t0,  # ← 新增
                "tree_stats": tree_stats,
            })

            hit_rate  = tree_stats.get("hit_rate", 0.0)
            attempts  = tree_stats.get("tree_attempts", 0)
            hits      = tree_stats.get("tree_hits", 0)
            avg_nodes = tree_stats.get("avg_nodes_per_tree", 0.0)

            _print_block(
                "Speculative + Tree Pre-Gen (新)", out_text, results["spec_tree"],
                color="magenta",
                extra_lines=[
                    f"Accept Rate : {accept_rate:.3f}",
                    f"Tree Hit    : {hits}/{attempts} ({hit_rate:.1%})",
                    f"Avg Nodes   : {avg_nodes:.1f}  "
                    f"(BF={self.branch_factor}, hi={self.high_thresh}, mid={self.mid_thresh})",
                ],
                dual_gpu=self._dual_gpu,
            )

        # ══════════════════ 4. Target 自回归（基线）══════════════════
        if self.run_target_ar:
            results["target"]["enabled"] = True
            self._set_seed(42)
            self._pm_start()
            t0 = time.time()

            output_ids = autoregressive_generate(
                tokenized,
                self.target,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )

            self._sync_all()
            t1 = time.time()
            self._pm_stop()

            te, de   = self._pm_energy_separate()
            out_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            results["target"].update({
                "output": out_text,
                "throughput": len(out_text) / (t1 - t0),
                "target_energy_j": te, "drafter_energy_j": de,
                "elapsed_s": t1 - t0,  # ← 新增
            })

            _print_block(
                "Target AR (基线)", out_text, results["target"],
                color="blue", dual_gpu=self._dual_gpu,
            )

        # ══════════════════ 5. Drafter 自回归 ══════════════════
        if self.run_drafter_ar:
            results["drafter"]["enabled"] = True
            self._set_seed(42)
            self._pm_start()
            t0 = time.time()

            output_ids = autoregressive_generate(
                tokenized,
                self.drafter,
                use_cache=self.cache,
                max_gen_len=self.gen_len,
                eos_tokens_id=self.end_tokens,
                logits_processor=self.processor,
                debug=self.debug,
            )

            self._sync_all()
            t1 = time.time()
            self._pm_stop()

            te, de   = self._pm_energy_separate()
            out_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            results["drafter"].update({
                "output": out_text,
                "throughput": len(out_text) / (t1 - t0),
                "target_energy_j": te, "drafter_energy_j": de,
                "elapsed_s": t1 - t0,  # ← 新增
            })

            _print_block(
                "Drafter AR", out_text, results["drafter"],
                color="cyan", dual_gpu=self._dual_gpu,
            )

        self._print_summary(results)

    # ─────────────────────────── 汇总打印 ───────────────────────────
    def _print_summary(self, results):
        W   = 120
        sep = "=" * W
        print("\n" + colored(sep, "white", attrs=["bold"]))
        print(colored("PERFORMANCE SUMMARY".center(W), "white", attrs=["bold"], on_color="on_grey"))
        print(colored(sep, "white", attrs=["bold"]))

        methods = [
            ("Speculative (标准)",        "spec",        "green"),
            ("Spec + Serial Pre-Gen",     "spec_pregen", "yellow"),
            ("Spec + Tree  Pre-Gen (新)", "spec_tree",   "magenta"),
            ("Target AR (基线)",           "target",      "blue"),
            ("Drafter AR",                "drafter",     "cyan"),
        ]

        base_tp      = results["target"]["throughput"] if results["target"]["enabled"] else 0
        base_elapsed = results["target"]["elapsed_s"]  if results["target"]["enabled"] else 0

        # ─── 1. 速度 & 能耗对比表 ───
        print(colored("\n📊 Speed & Energy Metrics:", "white", attrs=["bold"]))
        print(colored("-" * W, "white"))

        if self._dual_gpu:
            header = (
                f"{'Method':<32} {'Throughput':>14} {'Elapsed':>10} {'Accept':>8}"
                f" {'Target(J)':>11} {'Drafter(J)':>11} {'Total(J)':>10} {'vs Base':>9}"
            )
        else:
            header = (
                f"{'Method':<32} {'Throughput':>14} {'Elapsed':>10} {'Accept':>8}"
                f" {'Energy(J)':>11} {'vs Base':>9}"
            )

        print(colored(header, "white", attrs=["bold"]))
        print(colored("-" * W, "white"))

        for label, key, color in methods:
            r = results[key]
            if not r["enabled"]:
                continue

            tp_str      = f"{r['throughput']:.1f} chars/s"
            elapsed_s   = r.get("elapsed_s", 0.0)
            elapsed_str = f"{elapsed_s:.3f}s"
            ar_str      = f"{r.get('accept_rate', 0):.3f}" if r.get("accept_rate", 0) > 0 else "N/A"
            te          = r["target_energy_j"]
            de          = r["drafter_energy_j"]
            total_e     = te + de
            sp_str      = f"+{(r['throughput'] / base_tp - 1) * 100:.1f}%" if base_tp > 0 else "—"

            if self._dual_gpu:
                row = (
                    f"{label:<32} {tp_str:>14} {elapsed_str:>10} {ar_str:>8}"
                    f" {te:>10.4f}J {de:>10.4f}J {total_e:>9.4f}J {sp_str:>9}"
                )
            else:
                row = (
                    f"{label:<32} {tp_str:>14} {elapsed_str:>10} {ar_str:>8}"
                    f" {total_e:>10.4f}J {sp_str:>9}"
                )

            print(colored(row, color))

        # # ─── 2. 速度提升可视化 ───
        # if base_tp > 0:
        #     print(colored("\n⚡ Speedup vs Target AR:", "white", attrs=["bold"]))
        #     print(colored("-" * W, "white"))
        #     for label, key, color in methods:
        #         r = results[key]
        #         if not r["enabled"] or key == "target":
        #             continue
        #         speedup = r["throughput"] / base_tp
        #         bar_len = max(0, int(speedup * 15))
        #         bar     = "█" * bar_len
        #         print(colored(f"  {label:<32}: {speedup:5.2f}x  {bar}", color))

        # # ─── 3. 端到端时间对比 ───
        # if base_elapsed > 0:
        #     print(colored("\n⏱  End-to-End Time:", "white", attrs=["bold"]))
        #     print(colored("-" * W, "white"))
        #     print(colored(
        #         f"  {'Method':<32}  {'Elapsed':>10}  {'vs Target AR':>14}",
        #         "white", attrs=["bold"],
        #     ))
        #     for label, key, color in methods:
        #         r = results[key]
        #         if not r["enabled"]:
        #             continue
        #         elapsed_s = r.get("elapsed_s", 0.0)
        #         ratio     = elapsed_s / base_elapsed if base_elapsed > 0 else 1.0
        #         # ratio < 1 表示比基线快，用负号表示节省的比例
        #         sign      = "-" if ratio <= 1.0 else "+"
        #         pct       = abs(1.0 - ratio) * 100
        #         ratio_str = f"{sign}{pct:.1f}%"
        #         print(colored(
        #             f"  {label:<32}  {elapsed_s:>9.3f}s  {ratio_str:>14}",
        #             color,
        #         ))

        # # ─── 4. 能效分析 ───
        # print(colored("\n🔋 Energy Efficiency (chars/J):", "white", attrs=["bold"]))
        # print(colored("-" * W, "white"))

        # if self._dual_gpu:
        #     eff_header = (
        #         f"  {'Method':<32} {'chars/J(Target)':>16}"
        #         f" {'chars/J(Drafter)':>17} {'chars/J(Total)':>15}"
        #     )
        #     print(colored(eff_header, "white", attrs=["bold"]))
        # else:
        #     print(colored(f"  {'Method':<32} {'chars/J':>16}", "white", attrs=["bold"]))

        # for label, key, color in methods:
        #     r = results[key]
        #     if not r["enabled"] or not r["output"]:
        #         continue
        #     n_chars = len(r["output"])
        #     te, de  = r["target_energy_j"], r["drafter_energy_j"]
        #     total_e = te + de

        #     if self._dual_gpu:
        #         eff_t_s   = f"{n_chars / te:.2f}"      if te      > 0 else "N/A"
        #         eff_d_s   = f"{n_chars / de:.2f}"      if de      > 0 else "N/A"
        #         eff_tot_s = f"{n_chars / total_e:.2f}" if total_e > 0 else "N/A"
        #         print(colored(
        #             f"  {label:<32} {eff_t_s:>16} {eff_d_s:>17} {eff_tot_s:>15}",
        #             color,
        #         ))
        #     else:
        #         eff = n_chars / total_e if total_e > 0 else float("inf")
        #         print(colored(f"  {label:<32} {eff:>16.2f}", color))

        # # ─── 5. 树形预生成专属统计 ───
        # if results["spec_tree"]["enabled"]:
        #     ts = results["spec_tree"].get("tree_stats", {})
        #     print(colored("\n🌲 Tree Pre-Generation Stats:", "white", attrs=["bold"]))
        #     print(colored("-" * W, "white"))

        #     attempts   = ts.get("tree_attempts", 0)
        #     hits       = ts.get("tree_hits", 0)
        #     hit_rate   = ts.get("hit_rate", 0.0)
        #     tot_nodes  = ts.get("total_nodes", 0)
        #     tot_leaves = ts.get("total_leaves", 0)
        #     avg_nodes  = ts.get("avg_nodes_per_tree", 0.0)
        #     avg_leaves = ts.get("avg_leaves_per_tree", 0.0)

        #     print(colored(
        #         f"  Tree Pre-Gen Params   : BF={self.branch_factor},"
        #         f"  high_thresh={self.high_thresh}, mid_thresh={self.mid_thresh}",
        #         "magenta",
        #     ))
        #     print(colored(
        #         f"  Attempts / Hits       : {attempts} / {hits}  →  Hit Rate = {hit_rate:.1%}",
        #         "magenta",
        #     ))
        #     print(colored(
        #         f"  Total Nodes Explored  : {tot_nodes}  (avg {avg_nodes:.1f} / tree)",
        #         "magenta",
        #     ))
        #     print(colored(
        #         f"  Total Leaves          : {tot_leaves}  (avg {avg_leaves:.1f} / tree)",
        #         "magenta",
        #     ))

        #     if results["spec_pregen"]["enabled"] and results["spec_pregen"]["throughput"] > 0:
        #         pregen_tp = results["spec_pregen"]["throughput"]
        #         tree_tp   = results["spec_tree"]["throughput"]
        #         delta     = (tree_tp / pregen_tp - 1) * 100
        #         sign      = "+" if delta >= 0 else ""
        #         print(colored(
        #             f"  Tree vs Serial PreGen : {sign}{delta:.1f}%"
        #             f"  ({tree_tp:.1f} vs {pregen_tp:.1f} chars/s)",
        #             "magenta",
        #         ))

        # print(colored("\n" + "=" * W, "white", attrs=["bold"]))

    # ─────────────────────────── 运行所有场景 ───────────────────────────
    def _run(self):
        for scenario in test_scenarios:
            self._infer(scenario["name"], scenario["prompt"])
        self.power_monitor.shutdown()
        if self._dual_gpu and self.power_monitor_drafter:
            self.power_monitor_drafter.shutdown()

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─────────────────────────── 辅助打印 ───────────────────────────
def _print_block(
    title: str,
    text: str,
    result: dict,
    color: str,
    extra_lines=None,
    dual_gpu: bool = False,
):
    bar = f"{'='*10} {title} {'='*10}"
    print(colored(f"\n{bar}", color, attrs=["bold"]))
    print(colored("Output:", color), text)
    print(colored(f"Elapsed Time : {result.get('elapsed_s', 0.0):.3f} s", color))   # ← 新增
    print(colored(f"Throughput   : {result['throughput']:.1f} chars/s", color))

    te    = result.get("target_energy_j",  0.0)
    de    = result.get("drafter_energy_j", 0.0)
    total = te + de

    if dual_gpu:
        print(colored(f"Energy Target  ({InferenceCLI.TARGET_DEVICE})  : {te:.4f} J", color))
        print(colored(f"Energy Drafter ({InferenceCLI.DRAFTER_DEVICE}) : {de:.4f} J", color))
        print(colored(f"Energy Total                  : {total:.4f} J", color))
    else:
        print(colored(f"Energy       : {total:.4f} J", color))

    if extra_lines:
        for line in extra_lines:
            print(colored(line, color))
    print(colored("=" * len(bar), color, attrs=["bold"]))


# ─────────────────────────── 入口 ───────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(
        description="Speculative Decoding + Pre-Generation CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--target-device", type=str, default="cuda:6",
        help="Target 模型所在设备 (默认 cuda:1)",
    )
    parser.add_argument(
        "--drafter-device", type=str, default="cuda:7",
        help="Drafter 模型所在设备 (默认 cuda:2)",
    )
    parser.add_argument(
        "--high-thresh", type=float, default=0.8,
        help="高置信度阈值 (默认 0.8)",
    )
    parser.add_argument(
        "--mid-thresh", type=float, default=0.5,
        help="中置信度阈值 (默认 0.5)",
    )
    args = parser.parse_args()

    cli = InferenceCLI(
        target_device=args.target_device,
        drafter_device=args.drafter_device,
    )

    cli.high_thresh = args.high_thresh
    cli.mid_thresh  = args.mid_thresh