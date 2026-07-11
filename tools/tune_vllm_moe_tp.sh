#!/usr/bin/env bash
# TP-T3(docs/23/26): 为 vLLM triton fused_moe 生成 TP 形状的本机调优 config。
#
# TP 与 EP 的 config 键不同(microbench/tune_vllm_moe.sh 的 EP 产物救不了 TP):
#   - TP 下 fused_experts 见到全部 64 个 expert, intermediate 是分片 768
#     -> 查表键 E=64, N=768
#   - topk=8 全命中本地(无 EP 掩码), 展开行数 = M*8, 每 expert M/8 行
#     -> 直接用 topk=8 调优, 密度即真实口径, 无需 EP 的 topk 折算
#   - batch(M) = fused_experts 收到的 world*T: T∈{256,512,1024}/rank
#     -> M ∈ {1024, 2048, 4096}, 外加 512/8192 两档余量
#
# 用法(约 0.5~2 小时, 独占卡; 产物 json 装机后 serial 基线自动变快):
#   bash moe_bench/tools/tune_vllm_moe_tp.sh
set -euo pipefail

VLLM_DIR="${VLLM_DIR:-/data/cinnzhang_vllm_td_test/vllm_td-main}"
GPUS="${MB_GPUS:-${CUDA_VISIBLE_DEVICES:-9,11,13,15}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
fi

MODEL_DIR="$SCRIPT_DIR/build_tune/tune_model_tp"
mkdir -p "$MODEL_DIR"
cat > "$MODEL_DIR/config.json" <<EOF
{
  "architectures": ["MixtralForCausalLM"],
  "model_type": "mixtral",
  "hidden_size": 4096,
  "intermediate_size": 768,
  "num_local_experts": 64,
  "num_experts_per_tok": 8,
  "num_hidden_layers": 1,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 32000,
  "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-05,
  "torch_dtype": "bfloat16"
}
EOF
echo "[tune-tp] fake model config: $MODEL_DIR/config.json (E=64, N=768, topk=8)"

cd "$VLLM_DIR"
CUDA_VISIBLE_DEVICES="$GPUS" python benchmarks/kernels/benchmark_moe.py \
  --model "$MODEL_DIR" --tp-size 1 --dtype auto --seed 0 --tune \
  --batch-size 512 1024 2048 4096 8192

CFG_DIR="$VLLM_DIR/vllm/model_executor/layers/fused_moe/configs"
shopt -s nullglob
PRODUCED=(E=64,N=768*.json)
if [[ ${#PRODUCED[@]} -eq 0 ]]; then
  echo "[tune-tp] 未在 $VLLM_DIR 下找到 E=64,N=768*.json, 检查上面的输出" >&2
  exit 1
fi
for f in "${PRODUCED[@]}"; do
  cp -v "$f" "$CFG_DIR/"
  cp -v "$f" "$SCRIPT_DIR/build_tune/"
done
echo "[tune-tp] 完成。重跑 run_tp_all.sh, serial 基线将使用调优 config;"
echo "          验证点: serial 日志不再出现 'Using default MoE config' warning。"
