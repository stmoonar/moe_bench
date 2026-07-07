# Adding an efficient MoE implementation

How to plug a new MoE implementation into the benchmark so its numbers are
directly comparable to the baseline. For what the benchmark measures and how to
run it, see [`README.md`](README.md). There are two extension points:

- **Compute-only [`MoEImplementation`](baseline.py)** (`--impl`) — times just the
  local expert GEMMs on one rank. Use for single-GPU kernel comparisons. Covered
  next.
- **Distributed [`DistributedScheme`](schemes.py)** (`--scheme`) — one rank's
  *full* layer including dispatch/combine communication. Use for multi-GPU
  layers where comm/compute overlap or a fused layer op is the point. See
  [Distributed schemes](#distributed-schemes-full-layer-with-communication).

## The compute-only contract

The interface is [`MoEImplementation`](baseline.py). Every implementation
receives the **same** [`MoEProblem`](data.py) (inputs, weights, routing, quant
config) and returns the layer output `(num_tokens, hidden_size)`.

```python
# moe_bench/baseline.py
class MyFastMoE(MoEImplementation):
    name = "myfast"

    def setup(self, problem: MoEProblem) -> None:
        # One-time weight prep / kernel selection (not timed).
        ...

    def run(self, problem: MoEProblem) -> torch.Tensor:
        # Must be side-effect free & allocation-stable (CUDA-graph safe).
        return my_kernel(
            problem.hidden_states, problem.w1, problem.w2,
            problem.topk_weights, problem.topk_ids,
            expert_map=problem.expert_map,          # EP only
            quant_config=problem.quant_config,      # fp8 scales, or None
        )

IMPLEMENTATIONS["myfast"] = MyFastMoE
```

Then benchmark it against the baseline with the same config:

```bash
python -m moe_bench.bench --config moe_bench/configs/deepseek_v32.yaml --impl myfast
```

### `run` requirements

`run` is called repeatedly in the timing loop and may be captured in a CUDA
graph, so it must be:

- **side-effect free** — read only from `problem`, no dependence on Python state
  that changes between calls;
- **allocation-stable** — the same allocation pattern every call (allocating
  outputs/workspaces internally is fine; the graph's private pool owns them
  during capture).

Do one-time work (weight reshaping, kernel selection, autotuning) in `setup`,
which runs once per token count and is not timed.

## Correctness is checked automatically

Because every implementation sees byte-identical inputs, the harness verifies
your output against the same pure-torch reference it uses for the baseline — at
every token count, before reporting speed. You get a `verify` (`ok`/`FAIL`)
column for free; a mismatch prints the offending token count and a final
warning. See the [Correctness verification](README.md#correctness-verification)
section of the README for tolerances and how to tune them.

### Why the cross-check is sound (determinism)

Inputs are seeded from `(seed, num_tokens, rank)` and each expert-weight chunk
from `(seed, global_expert_id, shard)` — **not** from the rank — so:

- baseline and your implementation see byte-identical tensors at the same
  config → compare outputs directly before comparing speed;
- in EP, expert `e` has the same weights whoever owns it, so an EP run is
  checkable against a single-GPU reference over all experts;
- in TP, rank `r` always holds chunk `r` of every expert (the logical expert
  weight is the concatenation of per-shard chunks), so the all-reduced output
  is a reproducible forward of one well-defined model.

Weights are token-count independent and built once per sweep
(`make_weights(config, rank)`); pass them to `make_problem(..., weights=...)`
if you drive the harness programmatically.

## Distributed schemes (full layer with communication)

The `MoEImplementation` above is **compute-only**: `run` times just the local
expert GEMMs on one rank. To benchmark a *multi-GPU* MoE layer where
communication and computation are the whole point — dual-stream comm/compute
overlap, a fused MoE-layer op — use a **scheme** instead. A scheme owns one
rank's entire layer (dispatch + expert compute + combine) and returns that
rank's final output, so serial / overlap / fused strategies compare under one
harness.

### The scheme contract

The interface is [`DistributedScheme`](schemes.py). Every scheme takes the same
[`MoEProblem`](data.py) plus a [`DistContext`](context.py) (rank, world size,
device, process group) and returns this rank's `(num_tokens, hidden)` output
after combine.

```python
# moe_bench/schemes.py
class MyOverlapScheme(DistributedScheme):
    name = "overlap"

    def setup(self, problem: MoEProblem, ctx: DistContext) -> None:
        # Not timed: cache weights, allocate comm buffers / streams.
        self.ctx = ctx
        ...

    def run(self) -> torch.Tensor:
        # Timed, allocation-stable: full layer incl. comm on this rank.
        # e.g. split into micro-batches and overlap dispatch(mb2) with
        # expert-compute(mb1) on a second stream, then combine.
        ...

SCHEMES["overlap"] = MyOverlapScheme
```

Benchmark it against the serial baseline:

```bash
python -m moe_bench.bench --distributed --mode ep --precision fp8 \
    --world-size 8 --scheme overlap
```

### Unified I/O contract

Everything a scheme needs is on `problem`, identical across precisions — either
**FP8 + scales** or **BF16/FP16**:

| | input | weights | scales | routing |
|--|--|--|--|--|
| **BF16/FP16** | `hidden_states` (bf16) | `w1`, `w2` (bf16) | `quant_config is None` | `topk_ids`, `topk_weights` |
| **FP8 (w8a8)** | `hidden_states` (bf16; quantize internally) | `w1`, `w2` (fp8) | `quant_config` (w1/w2 block scales) | `topk_ids`, `topk_weights` |

Plus `problem.expert_map` (EP only: global→local expert id, `-1` if not on rank)
and `problem.config` for shapes. `hidden_states` is this rank's token shard
(`num_tokens` rows); the full batch is `world_size × num_tokens` after gather.
Output must be this rank's `num_tokens` rows after combine.

### The serial baseline

`SerialNaive` (`--scheme serial`, the default) is the reference dataflow with
**no** comm/compute overlap — write your overlap variant as a diff against it:

- **EP**: `AllGather` the token shard *and* its routing from every rank →
  `fused_experts(expert_map=…)` → `ReduceScatter` per-token results back.
- **TP**: `AllGather` the token shard → `fused_experts` with the
  intermediate-sharded weights (each rank a partial down-projection sum) →
  `ReduceScatter` (sums the partials, scatters each rank its tokens).

It uses plain `torch.distributed` collectives and vLLM's `fused_experts` — no
modular-kernel machinery — so the layer is transparent and easy to fork.

### Correctness

Distributed schemes are verified too (unless `--no-verify`): each rank's combined
output equals a full MoE over its own token shard, so it is checked against
`reference_moe` on that shard through the full logical weight set
(`make_golden_problem` / `make_logical_weights`). The weight generator is shared
between the sharded run and the reference (`_gen_expert_chunk`), so they are
consistent by construction. The fp8 reference loops over all experts and is slow
at large token counts — use `--no-verify` for pure timing runs.
