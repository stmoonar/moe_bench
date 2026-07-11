#!/usr/bin/env bash
# moe_bench microbench 一键运行:选卡 → mb1/2/4/5/6/7 → mb3 汇总分析。
# 结果(JSON + 日志 + 环境快照)保存到 microbench/results/<时间戳>/,
# 整个目录拷走即可做离线分析。
#
# 用法(任意目录):
#   bash moe_bench/microbench/run_all.sh
#
# 可选环境变量:
#   MB_GPUS=9,11,13,15     指定卡组(默认自动选第一组空闲的优先卡组)
#   MB_TOKENS=128,512      覆盖 token sweep(全 rank 总数,默认 7 档)
#   MB_ITERS / MB_WARMUP   计时迭代(默认 30 / 10)
#   MB_COMM_SMS=1,4,16     覆盖 mb5 的 comm SM 扫参
#   MB_PRECISION=fp8       精度(默认 bf16;fp8 需 TK scheme 支持,见 README)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # moe_bench 的上级目录
cd "$PARENT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
fi

# 选卡优先级:MB_GPUS > 已有的 CUDA_VISIBLE_DEVICES > 自动选空闲卡组。
# pick_gpus 用文件路径直跑(不走 -m):避免 import moe_bench 包(其 __init__
# 会连带 import vllm/torch/triton,在还没设 CUDA_VISIBLE_DEVICES 时可能因
# 看不到 GPU 而在 import 阶段炸 "0 active drivers")。
if [[ -n "${MB_GPUS:-}" ]]; then
  GPUS="$MB_GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPUS="$CUDA_VISIBLE_DEVICES"
else
  GPUS="$(python "$SCRIPT_DIR/pick_gpus.py")" || {
    echo "[run_all] 没有空闲的 4 卡组,退出(可用 MB_GPUS=... 强制指定)"
    exit 1
  }
fi
export CUDA_VISIBLE_DEVICES="$GPUS"
echo "[run_all] CUDA_VISIBLE_DEVICES=$GPUS"

# preflight:在设好 CUDA_VISIBLE_DEVICES 之后、跑任何测试之前,验证
# torch 能看到 >=4 张卡且 vllm 的 import 链(含定制 triton 的 driver 初始化)
# 走得通;失败时给出可操作的排查方向,而不是让 7 个测试各炸一遍。
if ! python - <<'EOF'
import torch
n = torch.cuda.device_count()
assert n >= 4, f"torch 只看到 {n} 张卡(<4)"
import vllm  # 触发 vllm -> torch._inductor -> triton driver 的完整 import 链
print(f"[preflight] torch={torch.__version__} cuda_devices={n} "
      f"vllm={vllm.__version__}")
EOF
then
  echo "[run_all] preflight 失败。排查顺序:"
  echo "  1) 当前 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 里的卡号是否有效/空闲;"
  echo "  2) 同一 shell 里裸跑: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -c 'import vllm'"
  echo "     若也报 '0 active drivers',说明是环境层问题(venv/torch/定制 triton),"
  echo "     与 microbench 无关 —— 此时 python -m moe_bench.bench 应同样失败;"
  echo "  3) venv 是否为 /data/cinnzhang_vllm_td_test/venvs/vllm-td(当前: ${VIRTUAL_ENV:-无})"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
export MB_OUT="$SCRIPT_DIR/results/$TS"
mkdir -p "$MB_OUT"
echo "[run_all] 结果目录: $MB_OUT"

# ---- 环境快照(可复现性) ----
nvidia-smi > "$MB_OUT/nvidia-smi.txt" 2>&1 || true
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv > "$MB_OUT/clocks.txt" 2>&1 || true
git -C "$SCRIPT_DIR/.." rev-parse HEAD > "$MB_OUT/git_commit.txt" 2>&1 || true
git -C "$SCRIPT_DIR/.." status --short >> "$MB_OUT/git_commit.txt" 2>&1 || true
python - > "$MB_OUT/env.txt" 2>&1 <<'EOF' || true
import sys, torch
print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:
    print("vllm import failed:", e)
EOF
env | grep -E '^(MB_|TK_|CUDA_|NCCL_)' > "$MB_OUT/env_vars.txt" 2>&1 || true

STATUS=0
run_step() {
  local name="$1"; shift
  echo
  echo "===== [$name] $* ====="
  local t0
  t0=$(date +%s)
  if "$@" 2>&1 | tee "$MB_OUT/${name}.log"; then
    echo "[$name] OK ($(( $(date +%s) - t0 ))s)"
  else
    echo "[$name] FAILED (继续跑其余测试)"
    STATUS=1
  fi
}

run_step mb1 python -m moe_bench.microbench.mb1_compute
run_step mb2 python -m moe_bench.microbench.mb2_comm
run_step mb4 python -m moe_bench.microbench.mb4_fusion
run_step mb5 python -m moe_bench.microbench.mb5_sm_sweep
run_step mb6 python -m moe_bench.microbench.mb6_inter_intra
run_step mb7 python -m moe_bench.microbench.mb7_pcie_schemes
# mb3 是纯后处理(stdlib),用文件路径直跑,不走 -m(不 import moe_bench 包)
run_step mb3 python "$SCRIPT_DIR/mb3_ratio.py" --results "$MB_OUT"

echo
echo "[run_all] 完成,结果目录: $MB_OUT"
ls -la "$MB_OUT"
exit $STATUS
