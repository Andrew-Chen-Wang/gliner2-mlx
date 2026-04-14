"""
Statistical benchmark: PyTorch CPU vs MLX GPU (Apple Silicon).

Runs identical extraction scenarios on both backends, measures latency,
and reports speedup with 95% confidence intervals and paired t-test p-values.

Usage:
  python benchmark_statistical.py --n 1000
"""

import argparse
import math
import random
import statistics
import time

import mlx.core as mx
import torch
from scipy import stats as sp_stats

# ─── Helpers ──────────────────────────────────────────────────────


def ci95(data):
    """95% CI half-width using t-distribution."""
    n = len(data)
    if n < 2:
        return 0.0
    se = statistics.stdev(data) / math.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    return t_crit * se


def paired_test(old_times, new_times):
    """Paired t-test on matched samples."""
    diffs = [o - n for o, n in zip(old_times, new_times, strict=True)]
    n = len(diffs)
    mean_d = statistics.mean(diffs)
    se_d = statistics.stdev(diffs) / math.sqrt(n)
    t_stat = mean_d / se_d if se_d > 0 else 0
    p_val = 2 * sp_stats.t.sf(abs(t_stat), df=n - 1)
    hw = ci95(diffs)
    return t_stat, p_val, mean_d, hw


def fmt_p(p):
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def sync_torch():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sync_mlx():
    mx.eval(mx.array(0))


# ─── Models ──────────────────────────────────────────────────────


def load_models(model_id):
    """Load both PyTorch (CPU) and MLX models."""
    print("Loading PyTorch model (CPU)...", flush=True)
    from gliner2.inference.engine import GLiNER2

    pt_model = GLiNER2.from_pretrained(model_id, map_location="cpu")
    pt_model.eval()

    print("Loading MLX model (GPU)...", flush=True)
    from gliner2_mlx import GLiNER2MLX

    mlx_model = GLiNER2MLX.from_pretrained(model_id)

    return pt_model, mlx_model


# ─── Benchmark runner ────────────────────────────────────────────


def interleaved(cpu_fn, gpu_fn, n_warmup, n_iter):
    """Run CPU/GPU interleaved to eliminate ordering effects. Returns paired lists."""
    for _ in range(n_warmup):
        cpu_fn()
        gpu_fn()
    sync_torch()
    sync_mlx()

    cpu_times = []
    gpu_times = []
    for _ in range(n_iter):
        if random.random() < 0.5:
            sync_torch()
            t0 = time.perf_counter()
            cpu_fn()
            sync_torch()
            cpu_times.append((time.perf_counter() - t0) * 1000)

            sync_mlx()
            t0 = time.perf_counter()
            gpu_fn()
            sync_mlx()
            gpu_times.append((time.perf_counter() - t0) * 1000)
        else:
            sync_mlx()
            t0 = time.perf_counter()
            gpu_fn()
            sync_mlx()
            gpu_times.append((time.perf_counter() - t0) * 1000)

            sync_torch()
            t0 = time.perf_counter()
            cpu_fn()
            sync_torch()
            cpu_times.append((time.perf_counter() - t0) * 1000)

    return cpu_times, gpu_times


def print_paired(cpu_t, gpu_t):
    m_cpu, m_gpu = statistics.mean(cpu_t), statistics.mean(gpu_t)
    ci_cpu, ci_gpu = ci95(cpu_t), ci95(gpu_t)
    _t_stat, p_val, _mean_diff, _hw = paired_test(cpu_t, gpu_t)
    speedup = m_cpu / m_gpu if m_gpu > 0 else float("inf")

    sig = ""
    if p_val < 0.05:
        sig = "*"
    if p_val < 0.01:
        sig = "**"
    if p_val < 0.001:
        sig = "***"

    print(
        f"  CPU: {m_cpu:>8.2f} ± {ci_cpu:.2f} ms  |  "
        f"MLX: {m_gpu:>8.2f} ± {ci_gpu:.2f} ms  |  "
        f"Speedup: {speedup:.2f}x  |  "
        f"p={fmt_p(p_val)}{sig}"
    )


# ─── Scenarios ───────────────────────────────────────────────────


def run_benchmark(pt_model, mlx_model, n_iter, n_warmup):
    """Run all scenarios and print results."""

    text_short = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12, 2023."
    ents = ["company", "person", "product", "location", "date"]

    texts_batch = [
        "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino.",
        "Google's Sundar Pichai spoke at the conference in Mountain View.",
        "Microsoft released Windows 11 in Redmond last year.",
        "Amazon founder Jeff Bezos invested in Blue Origin in Seattle.",
        "Tesla CEO Elon Musk unveiled the Cybertruck at the Fremont factory.",
        "Meta's Mark Zuckerberg presented Quest 3 in Menlo Park.",
        "NVIDIA's Jensen Huang showcased the H100 GPU at GTC in San Jose.",
        "OpenAI CEO Sam Altman launched GPT-4 in San Francisco.",
    ]

    long_text = (
        "Apple Inc., headquartered in Cupertino, California, is a multinational technology company "
        "founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company designs, "
        "develops, and sells consumer electronics, computer software, and online services. Tim Cook "
        "has served as CEO since August 2011. Apple's main products include the iPhone, iPad, Mac, "
        "Apple Watch, and AirPods. The company also operates services including the App Store, "
        "Apple Music, iCloud, and Apple TV Plus. In 2023, Apple reported annual revenue of $383 "
        "billion, making it the world's largest technology company by revenue. The company employs "
        "over 160,000 people worldwide."
    )
    ents6 = ["company", "person", "product", "location", "date", "monetary_value"]

    text_struct = "John Smith, aged 35, is a software engineer at Google in Mountain View."
    pt_schema_struct = pt_model.create_schema()
    pt_schema_struct.structure("person").field("name").field("age").field("job_title").field("company").field(
        "location"
    )
    from gliner2.inference.engine import Schema

    mlx_schema_struct = Schema()
    mlx_schema_struct.structure("person").field("name").field("age").field("job_title").field("company").field(
        "location"
    )

    text_rel = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12."
    rels = ["CEO_of", "located_in", "announced_on"]

    scenarios = [
        (
            "Single entity extraction",
            lambda: pt_model.extract_entities(text_short, ents),
            lambda: mlx_model.extract_entities(text_short, ents),
        ),
        (
            "Batch entity extraction (8)",
            lambda: pt_model.batch_extract_entities(texts_batch, ents, batch_size=8),
            lambda: mlx_model.batch_extract_entities(texts_batch, ents, batch_size=8),
        ),
        (
            "Long text entity extraction",
            lambda: pt_model.extract_entities(long_text, ents6),
            lambda: mlx_model.extract_entities(long_text, ents6),
        ),
        (
            "Structure extraction",
            lambda: pt_model.extract(text_struct, pt_schema_struct),
            lambda: mlx_model.extract(text_struct, mlx_schema_struct),
        ),
        (
            "Relation extraction",
            lambda: pt_model.extract_relations(text_rel, rels),
            lambda: mlx_model.extract_relations(text_rel, rels),
        ),
    ]

    print(f"\n{'=' * 90}")
    print(f"  PyTorch CPU vs MLX GPU  |  {n_iter} iterations  |  {n_warmup} warmup")
    print(f"{'=' * 90}\n")

    all_cpu = []
    all_gpu = []

    for name, cpu_fn, gpu_fn in scenarios:
        print(f"[{name}]")
        with torch.inference_mode():
            cpu_t, gpu_t = interleaved(cpu_fn, gpu_fn, n_warmup, n_iter)
        print_paired(cpu_t, gpu_t)
        all_cpu.extend(cpu_t)
        all_gpu.extend(gpu_t)
        print()

    # Overall summary
    m_cpu, m_gpu = statistics.mean(all_cpu), statistics.mean(all_gpu)
    overall_speedup = m_cpu / m_gpu if m_gpu > 0 else float("inf")
    print(f"{'=' * 90}")
    print(f"  Overall mean:  CPU {m_cpu:.2f} ms  |  MLX {m_gpu:.2f} ms  |  Speedup: {overall_speedup:.2f}x")
    print(f"{'=' * 90}")


# ─── Main ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch CPU vs MLX GPU")
    parser.add_argument("--n", type=int, default=1000, help="Iterations per scenario")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument(
        "--model",
        default="fastino/gliner2-base-v1",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()

    pt_model, mlx_model = load_models(args.model)
    run_benchmark(pt_model, mlx_model, args.n, args.warmup)


if __name__ == "__main__":
    main()
