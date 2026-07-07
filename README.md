# MoE Layer Benchmark

A config-driven micro-benchmark for vLLM's MoE (Mixture-of-Experts) layer.

The **baseline** is vLLM's *naive* fused-MoE — the same Triton `fused_experts`
path the unquantized/native MoE method dispatches to. Future efficient
implementations plug into the same harness (identical inputs, identical output
contract) so their numbers are directly comparable.

Defaults match **DeepSeek V3.2** (`hidden=7168`, `intermediate=2048`,
`experts=256`, `topk=8`).

## Two modes

| Mode | `distributed` | GPUs | Measures |
|------|---------------|------|----------|
| **Compute-only** (default) | `false` | 1 | just the local expert GEMMs (`fused_experts`) on one rank — a pure kernel benchmark. |
| **Distributed** | `true` | `world_size` | the **full MoE layer including communication**: TP output all-reduce, or EP naive all-gather-dispatch + reduce-scatter-combine. |

Compute-only is for quick single-GPU kernel comparisons. Distributed is for
end-to-end TP-vs-EP layer performance where communication matters (often the
EP bottleneck). See [Distributed mode](#distributed-mode-communication) below.

## What it measures

For each token count in the sweep, it times one MoE forward
(`x @ w1 → SiLU → @ w2 → weighted-sum`) and reports:

- **latency (µs)** — average wall time per forward. In compute-only mode the
  op is captured in a CUDA graph (10 invocations per graph, warmed after
  capture, launch overhead amortized out — same methodology as
  `benchmarks/kernels/benchmark_moe.py`).
- **TFLOP/s** — achieved throughput of the two per-expert GEMMs
  (`6·N·K` flops per computed token-expert assignment).
- **local_assignments** — number of token-expert assignments actually computed
  on this rank (all of them in TP; the local share in EP). In distributed mode
  this is the max across ranks, consistent with latency being gated by the
  slowest rank.
- **timing** — `graph` or `eager`. If CUDA-graph capture fails the point falls
  back to eager timing; the column (also saved to JSON) makes mixed-regime
  results visible instead of silently comparable.
- **rel_err / verify** — correctness cross-check against a pure-torch reference
  (on by default). See [Correctness verification](#correctness-verification).

## Correctness verification

Before timing each token count, the harness computes the layer output with a
pure-torch reference ([`reference.py`](reference.py)) built from the *same*
problem inputs, then — after the timing loop — runs one more untimed forward and
compares it with `torch.testing.assert_close` (the same check vLLM's own
fused-MoE tests use). Two extra columns are reported (and saved to JSON):

- **rel_err** — relative L2 error `‖out − ref‖ / ‖ref‖`, an at-a-glance metric.
- **verify** — `ok` / `FAIL` from the elementwise `assert_close`. On failure the
  per-token diagnostic (mismatch count + worst element) is printed and a final
  warning is emitted, but the sweep still finishes.

The reference matches the kernel's numerics per precision: bf16/fp16 accumulate
the dense per-expert GEMMs in fp32; fp8 block-quantizes the activations exactly
as the kernel does and runs a native block matmul, so both carry the same
quantization error and stay comparable within a tight tolerance. Tolerances
default per precision (`assert_close(atol, rtol)`, mirroring
`tests/kernels/moe`); override with `--verify-atol` / `--verify-rtol` or the
`verify_atol` / `verify_rtol` config fields.

This validates the baseline *and* any efficient implementation you add (both see
byte-identical inputs → both are checked against the same ground truth). Disable
with `--no-verify` (or `verify: false`) for pure timing runs. The fp8 reference
(a native block matmul looping over experts × tiles) is slow at large token
counts — `--no-verify` skips it when you only want timings.

> Distributed schemes are verified too: each rank's combined output equals a full
> MoE over its own token shard, checked against `reference_moe` on that shard
> through the full logical weights (`make_golden_problem`). The fp8 reference is
> slow at large token counts — use `--no-verify` for pure timing runs.

```bash
# Default run already verifies; tighten the fp8 tolerance:
python -m moe_bench.bench --precision fp8 --verify-atol 0.02 --verify-rtol 0.02

# Skip verification for a pure timing pass:
python -m moe_bench.bench --no-verify
```

## Requirements

A CUDA device and a working vLLM install (the benchmark imports
`vllm.model_executor.layers.fused_moe`). Follow the environment setup in the
repo's `AGENTS.md` (`uv venv`, `uv pip install -e .`).

## Quick start

```bash
# Baseline, DeepSeek V3.2 defaults (TP=8, bf16):
python -m moe_bench.bench --config moe_bench/configs/deepseek_v32.yaml

# Expert-parallel, fp8 block-quant:
python -m moe_bench.bench --config moe_bench/configs/deepseek_v32.yaml \
    --mode ep --precision fp8

# Stress load imbalance (all tokens to the same experts):
python -m moe_bench.bench --distribution single

# Custom token sweep + save results:
python -m moe_bench.bench --num-tokens 1 8 128 2048 --output-json out.json
```

Run without `--config` to use the built-in DeepSeek V3.2 defaults.

## Parallelism: TP vs EP

The benchmark runs on a single device but reproduces the *local* work of one
rank in a group of `world_size`:

| Mode | Local experts        | Intermediate dim        | Routing |
|------|----------------------|-------------------------|---------|
| `tp` | `num_experts` (all)  | `intermediate / world`  | over all experts, no `expert_map` |
| `ep` | `num_experts / world`| `intermediate` (full)   | over all experts; non-local masked via `expert_map` |

In EP, tokens are routed over the *global* expert set and an `expert_map` masks
out experts that don't live on this rank, so only the local share of the routed
tokens is computed — mirroring what a rank sees after all-to-all dispatch.

## Distributed mode (communication)

Set `distributed: true` (or pass `--distributed`) to spawn `world_size`
single-GPU processes and measure the **full layer including cross-rank
communication**. The unit under benchmark is a **scheme** — one rank's whole
layer (dispatch + expert compute + combine) — selected with `--scheme`. The
default `serial` baseline runs comm and compute serially with plain
`torch.distributed` collectives + vLLM's `fused_experts`:

- **EP**: **AllGather** the token shard + routing → `fused_experts(expert_map)`
  → **ReduceScatter** the per-token results back to each rank.
- **TP**: **AllGather** the token shard → `fused_experts` with intermediate-
  sharded weights (partial down-projection sums) → **ReduceScatter** (sums the
  partials, scatters each rank its tokens).

```bash
# 8-GPU EP layer benchmark (fp8), serial baseline:
python -m moe_bench.bench --config moe_bench/configs/deepseek_v32.yaml \
    --distributed --mode ep --precision fp8 --world-size 8

# 8-GPU TP layer benchmark (bf16):
python -m moe_bench.bench --distributed --mode tp --world-size 8

# your own overlap / fused-layer scheme:
python -m moe_bench.bench --distributed --mode ep --world-size 8 --scheme overlap
```

The harness launches the processes itself (via `torch.multiprocessing.spawn`),
so run it as a normal `python -m moe_bench.bench` command — no `torchrun`
needed. Reported latency is the **max across ranks** (avg/min/med per call; the
slowest rank gates the step). CUDA graphs are disabled in this mode. Requires
`world_size` visible GPUs.

> Compute-only mode reports TFLOP/s; distributed mode reports layer latency
> (communication has no meaningful FLOP count to attribute).

The serial baseline is the reference dataflow; comm/compute **overlap** and
**fused-layer** schemes plug into the same harness and I/O contract. See
[`ADDING_IMPLEMENTATIONS.md`](ADDING_IMPLEMENTATIONS.md) for the
`DistributedScheme` interface.

## Configuration

All knobs live in a YAML file (see
[`configs/deepseek_v32.yaml`](configs/deepseek_v32.yaml)). CLI flags override
the file. Key fields:

| Field | Meaning |
|-------|---------|
| `hidden_size`, `intermediate_size`, `num_experts`, `topk` | model/weight shape |
| `parallel_mode` | `tp` or `ep` |
| `world_size` | shards `intermediate_size` (TP) or `num_experts` (EP) |
| `precision` | `bf16`, `fp16`, or `fp8` (per-block w8a8) |
| `block_shape` | fp8 scale grid, e.g. `[128, 128]` |
| `num_tokens` | list of token counts to sweep |
| `routing.distribution` | `balanced`, `uniform`, `skewed`, `single` |
| `routing.skew_alpha` | Zipf exponent for `skewed` |
| `routing.num_active_experts` | restrict routing to the first N experts |
| `distributed` | `true` = spawn `world_size` procs + measure comm; `false` = single-GPU compute-only |
| `all2all_backend` | EP dispatch/combine backend (`allgather_reducescatter` = naive AllGather+ReduceScatter baseline) |
| `warmup_iters`, `bench_iters`, `use_cuda_graph`, `seed` | timing |
| `verify`, `verify_atol`, `verify_rtol` | torch-reference correctness check (see [Correctness verification](#correctness-verification)) |

### Routing distributions

- `balanced` — every (active) expert gets equal token load. **Default.**
- `uniform` — each token picks `topk` distinct experts uniformly at random.
- `skewed` — Zipf-like concentration on a few hot experts (`skew_alpha`).
- `single` — every token routes to experts `[0, topk)`; maximal imbalance.

Precision applies to both activations and weights: `bf16`/`fp16` run the dense
GEMMs directly; `fp8` block-quantizes the weights (with matching scales) and
uses the w8a8 fused path.

## Adding an efficient implementation

Implementations register against the [`MoEImplementation`](baseline.py) contract
and are selected with `--impl`. Every one receives the same
[`MoEProblem`](data.py) and is correctness-checked against the shared torch
reference for free. See **[`ADDING_IMPLEMENTATIONS.md`](ADDING_IMPLEMENTATIONS.md)**
for the interface, the `run`/`setup` requirements, and how to extend the
distributed path.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Config schema + YAML loader; shape/precision/routing/timing |
| `data.py` | Deterministic input/weight/routing generation → `MoEWeights` / `MoEProblem` |
| `baseline.py` | `MoEImplementation` interface + naive `fused_experts` baseline (compute-only) |
| `reference.py` | Pure-torch ground-truth MoE forward + `assert_close` correctness check |
| `bench.py` | Timing harness (CUDA-graph batched capture), token sweep, CLI |
| `context.py` | `DistContext` — per-rank parallel handles for distributed schemes |
| `schemes.py` | `DistributedScheme` interface + serial full-layer baseline (comm + compute) |
| `distributed.py` | Multi-GPU harness: spawns ranks, runs a scheme, times + verifies full layer |
| `report.py` | Shared table/JSON reporting for both modes |
| `configs/deepseek_v32.yaml` | Default config (DeepSeek V3.2) |
| `ADDING_IMPLEMENTATIONS.md` | How to plug in and correctness-check a new MoE kernel |
