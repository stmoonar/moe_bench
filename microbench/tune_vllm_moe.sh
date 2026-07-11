#!/usr/bin/env bash
# P3(docs/19):为 vLLM triton fused_moe 生成本机调优 config。
#
# 背景:mb1 显示 vLLM 在用默认 MoE config(warning: Config file not found at
# .../configs/E=16,N=3072,device_name=NVIDIA_RTX_PRO_5000_72GB_Blackwell.json),
# 即基线未调优,"TK 计算更快"的结论需要在调优后复测。
#
# benchmark_moe.py 的 --model 只读 config.json 推形状(E/intermediate/hidden/
# topk/dtype),不加载权重 —— 所以用一个只含 config.json 的假 Mixtral 目录即可,
# 不需要真实模型。
#
# 形状映射(我们的单层 EP 配置 -> 假模型 config):
#   json 文件名里的 E=16 是 kernel 看到的本地专家数(E_global 64 / world 4)
#     -> num_local_experts = 16,--tp-size 1(不能用 64:EP 掩码后 kernel 只见 16)
#   N=3072 是 intermediate 分片 -> intermediate_size = 3072
#   hidden 4096,bf16。
#
# topk 的选择(默认 2,可传参 8):
#   运行时 fused_experts 收到 M 个 gathered token、topk=8 路由在 64 个全局专家上,
#   本地只命中 1/4 -> 实际展开行数 = M*8/4 = M*2,每本地专家 M/8 行。
#   config 查表键只有 (E, N, dtype, M),不含 topk —— 用 topk=2、16 专家均匀路由
#   调优,每专家行数与真实负载完全一致(密度匹配)。想按"标准"口径(topk=8 全
#   命中)调优则 `bash tune_vllm_moe.sh 8`,两者都调完可用 mb1 选更快的那份。
#
# 用法(调优约 0.5~2 小时,Ray 会并行用满 CUDA_VISIBLE_DEVICES 里的卡):
#   bash moe_bench/microbench/tune_vllm_moe.sh [topk]
# 产物 json 自动拷入 vllm 源码的 configs/ 目录;之后重跑 mb1+mb4+mb3 复测,
# warning 应消失。
set -euo pipefail

TOPK="${1:-2}"
VLLM_DIR="${VLLM_DIR:-/data/cinnzhang_vllm_td_test/vllm_td-main}"
GPUS="${MB_GPUS:-9,11,13,15}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate
fi

# ---- 假模型目录(只有 config.json,Mixtral 架构字段) ----
MODEL_DIR="$SCRIPT_DIR/build/tune_model_topk${TOPK}"
mkdir -p "$MODEL_DIR"
cat > "$MODEL_DIR/config.json" <<EOF
{
  "architectures": ["MixtralForCausalLM"],
  "model_type": "mixtral",
  "hidden_size": 4096,
  "intermediate_size": 3072,
  "num_local_experts": 16,
  "num_experts_per_tok": ${TOPK},
  "num_hidden_layers": 1,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 32000,
  "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-05,
  "torch_dtype": "bfloat16"
}
EOF
echo "[tune] fake model config: $MODEL_DIR/config.json (topk=$TOPK)"

# ---- 调优(在 vllm 根目录跑;产物 json 落在当前目录) ----
# batch-size 用我们 bench 的 M 档(fused_experts 收到的 gathered 总 token 数)。
# 若你们 vllm 版本的 --batch-size 不接受多值,请查 --help 后逐档跑或去掉该参数
# (默认 sweep 1..4096,运行时按最近 M 取 config,也可用)。
cd "$VLLM_DIR"
CUDA_VISIBLE_DEVICES="$GPUS" python benchmarks/kernels/benchmark_moe.py \
  --model "$MODEL_DIR" --tp-size 1 --dtype auto --seed 0 --tune \
  --batch-size 128 512 1024 2048 5120 6648 8192

# ---- 安装产物 ----
CFG_DIR="$VLLM_DIR/vllm/model_executor/layers/fused_moe/configs"
shopt -s nullglob
PRODUCED=(E=16,N=3072*.json)
if [[ ${#PRODUCED[@]} -eq 0 ]]; then
  echo "[tune] 未在 $VLLM_DIR 下找到 E=16,N=3072*.json,检查上面的输出" >&2
  exit 1
fi
for f in "${PRODUCED[@]}"; do
  cp -v "$f" "$CFG_DIR/"
  cp -v "$f" "$SCRIPT_DIR/build/"   # 留一份备份在 microbench/build/
done
echo "[tune] 完成。复测:bash moe_bench/microbench/run_all.sh"
echo "       验证点:mb1 日志不再出现 'Using default MoE config' warning;"
echo "       对比 vllm_compute_ms 前后差异,更新 docs/19 §0 的收益分解结论。"
