#!/usr/bin/env bash
# TP-T3 v2(docs/28): 为 vLLM triton fused_moe 生成 TP 形状的本机调优 config。
#
# 第八轮教训: v1 走 vllm benchmarks/kernels/benchmark_moe.py --tune, 其 ray
# 在本机(256 核共享机)卡死在 CoreWorker RegisterClient 2.5 小时, 被 timeout
# SIGTERM, 一个 trial 都没跑。v2 换成 tune_moe_tp_noray.py(纯 subprocess
# 多卡分片, 无 ray), 并且先跑 <2min 的 smoke 自检, 注入无效立刻失败退出,
# 不再白烧几个小时。
#
# TP 与 EP 的 config 键不同(EP 产物救不了 TP):
#   - TP 下 fused_experts 见到全部 E 个 expert, intermediate 是分片 768
#     -> 查表键 E=<NE>,N=768
#   - topk=8 全命中本地, batch(M) = world*T ∈ {1024,2048,4096}, 外加余量
#
# 默认只调主报数形状 E=64(NE=64/topk=8/hidden=4096/gate_up=6144);
# NE sweep 档位需要时用 TUNE_E="64 128 256" 打开。
#
# 用法(单档 E 约 15~20 分钟, 4 卡并行; 产物 json 装机后 serial 基线自动变快):
#   bash moe_bench/tools/tune_vllm_moe_tp.sh
set -euo pipefail

GPUS="${MB_GPUS:-${CUDA_VISIBLE_DEVICES:-9,11,13,15}}"
TUNE_E="${TUNE_E:-64}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
fi

echo "[tune-tp] smoke 自检(单卡, <2min): 验证注入机制在本机 vllm 上生效"
python "$SCRIPT_DIR/tune_moe_tp_noray.py" --smoke --gpus "$GPUS" --num-experts 64

for E in $TUNE_E; do
  echo "[tune-tp] ===== E=$E, N=768 ====="
  python "$SCRIPT_DIR/tune_moe_tp_noray.py" --gpus "$GPUS" --num-experts "$E"
done

echo "[tune-tp] 完成。重跑 serial, 基线将使用调优 config;"
echo "          验证点: serial 日志不再出现 'Using default MoE config' warning。"
