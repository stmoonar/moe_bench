#!/usr/bin/env bash
# probe_engine.sh — TK fp8 GEMM 引擎归因探针一键脚本（单卡），自动打包产物。
#
# 跑 5 组测试（任一组失败不中断后续，结果记入 STATUS.txt）：
#   P0  干净编译 + 基线 verify（L0/L1 形状 ×REPEATS）+ .so syscall 普查/资源占用
#   P1  TMA 试金石（tools/tma_litmus.cu，编译期，不占 GPU）
#   P2  K 扫描（N=4096，K∈{768,1536,3072,4096}，定价每任务固定开销）
#   P3  NCU 归因（L0/L1 × fp8/raw 共 4 份 .ncu-rep，锁频口径，--no-ncu 跳过）
#   P4  TASK_Q 2→4 A/B（默认跳过，--taskq4 开启：sed 改正本头文件→测→还原→重建基线 .so）
#
# 用法（GPU 机器，任意目录）：
#   bash moe_bench/tools/probe_engine.sh <GPU_ID> [--no-ncu] [--taskq4] [--skip-build] [--repeats N]
# 产物目录 + 压缩包：moe_bench/tp_test_results/probe_<时间戳>_engine{,.zip|.tar.gz}
#
# 口径说明：verify_fp8_gemm 的两个形状 = 主配置 rank0 视角（docs/08 §1），
# 本脚本是单卡引擎探针，不是 benchmark，不走通用 bench 入口；NCU 保持默认
# clock-control（锁基频），与 docs/08/10 的 789/664µs 锚点同口径。
set -u

# ---------- 参数 ----------
GPU="${1:-}"
[ -z "$GPU" ] && { echo "用法: bash moe_bench/tools/probe_engine.sh <GPU_ID> [--no-ncu] [--taskq4] [--skip-build] [--repeats N]"; exit 1; }
GPU="${GPU%%,*}"     # 单卡探针: 传了卡组也只取第一张(空闲检查/计时都按单卡口径)
shift
DO_NCU=1; DO_TASKQ=0; SKIP_BUILD=0; REPEATS=3
while [ $# -gt 0 ]; do
  case "$1" in
    --no-ncu)     DO_NCU=0 ;;
    --taskq4)     DO_TASKQ=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --repeats)    REPEATS="$2"; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
  shift
done

PYTHON="${PYTHON:-/root/miniconda3/envs/vllm-td/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$SCRIPT_DIR/../.." && pwd)"          # moe_bench 的上一级（/workspace）
cd "$WS"
[ -d moe_bench ] || { echo "错误: $WS 下没有 moe_bench/"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
OUT="moe_bench/tp_test_results/probe_${TS}_engine"
mkdir -p "$OUT"
STATUS="$OUT/STATUS.txt"; : > "$STATUS"
exec > >(tee "$OUT/probe_console.log") 2>&1

SO="moe_bench/kernels/tk/build/tk_moe_w4_h4096_rb128.so"
HDR="moe_bench/kernels/tileoverlap/common/sm120_common.cuh"

note() { echo "[probe $(date +%H:%M:%S)] $*"; }
mark() { echo "$1: $2" >> "$STATUS"; note "$1 -> $2"; }

# ---------- 前置检查 ----------
note "输出目录: $OUT"
"$PYTHON" -c "import torch" 2>/dev/null || { echo "错误: $PYTHON 里没有 torch（PYTHON 环境变量可覆盖）"; exit 1; }
command -v nvcc >/dev/null || { echo "错误: 找不到 nvcc"; exit 1; }

# 空闲卡红线（AGENTS.md）：有任务/利用率高就拒跑
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" | grep -c . || true)
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
if [ "$BUSY" -gt 0 ] || [ "${UTIL:-100}" -gt 5 ]; then
  echo "错误: GPU $GPU 不空闲（进程数=$BUSY, util=${UTIL}%），换一张卡"; exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU"

# 环境指纹（波动取证要求）
{ nvidia-smi
  nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.mem,temperature.gpu --format=csv
} > "$OUT/env_nvidia_smi.txt" 2>&1
nvcc --version > "$OUT/env_nvcc.txt" 2>&1
{ git -C moe_bench rev-parse HEAD; git -C moe_bench status --short; } > "$OUT/env_git.txt" 2>&1
echo "GPU=$GPU REPEATS=$REPEATS DO_NCU=$DO_NCU DO_TASKQ=$DO_TASKQ SKIP_BUILD=$SKIP_BUILD PYTHON=$PYTHON" > "$OUT/env_args.txt"

run_verify_set() {  # $1=输出文件；L0/L1 两形状各 REPEATS 遍
  local f="$1" i
  for i in $(seq 1 "$REPEATS"); do
    echo "===== round $i / L0 (K=4096 N=1536) ====="
    "$PYTHON" -m moe_bench.tools.verify_fp8_gemm 64 256 4096 1536 20
    echo "===== round $i / L1 (K=768 N=4096) ====="
    "$PYTHON" -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096 20
  done > "$f" 2>&1
}

# ---------- P0: 干净编译 + 基线 + syscall 普查 ----------
p0() {
  if [ "$SKIP_BUILD" -eq 0 ] || [ ! -f "$SO" ]; then
    note "P0: 干净编译（rm -rf build）"
    rm -rf moe_bench/kernels/tk/build
    "$PYTHON" moe_bench/kernels/tk/build.py 4 > "$OUT/build_log.txt" 2>&1 || return 1
  else
    note "P0: --skip-build，复用现有 .so"
    echo "(skip-build: 复用已有 .so)" > "$OUT/build_log.txt"
  fi
  [ -f "$SO" ] || return 1
  note "P0: 基线 verify（两形状 ×$REPEATS，boost 口径）"
  run_verify_set "$OUT/baseline_verify.txt" || return 1
  cuobjdump -sass "$SO" | grep -E 'CALL\.ABS|__cuda_syscall' | sort | uniq -c \
      > "$OUT/so_syscall_census.txt" 2>&1
  cuobjdump -res-usage "$SO" > "$OUT/so_res_usage.txt" 2>&1
  grep -E 'FAIL' "$OUT/baseline_verify.txt" && return 1
  return 0
}

# ---------- P1: TMA 试金石（编译期） ----------
p1() {
  note "P1: 编译 tma_litmus.cu"
  local SRC="moe_bench/tools/tma_litmus.cu" CUBIN="$OUT/tma_litmus.cubin"
  if ! nvcc -arch=sm_120a -cubin --ptxas-options=-v -o "$CUBIN" "$SRC" \
       2> "$OUT/tma_litmus.ptxas"; then
    note "P1: 直编失败，加 -DNO_CTA_DST 重试（cta 目标不可用本身就是结论）"
    echo "NO_CTA_DST=1（.shared::cta 目标 load 编译不过）" > "$OUT/tma_litmus_cta_unsupported.txt"
    nvcc -arch=sm_120a -cubin --ptxas-options=-v -DNO_CTA_DST -o "$CUBIN" "$SRC" \
         2>> "$OUT/tma_litmus.ptxas" || return 1
  fi
  cuobjdump -sass "$CUBIN"      > "$OUT/tma_litmus.sass" 2>&1
  cuobjdump -res-usage "$CUBIN" > "$OUT/tma_litmus.res"  2>&1
  awk '/Function :/ {f=$3} /CALL\.ABS|__cuda_syscall|UTMALDG|UTMASTG|UBLKCP/ {print f, $0}' \
      "$OUT/tma_litmus.sass" > "$OUT/tma_litmus_verdict.txt"
  return 0
}

# ---------- P2: K 扫描（每任务固定开销定价） ----------
p2() {
  note "P2: K 扫描 N=4096, K in 768/1536/3072/4096"
  local K
  for K in 768 1536 3072 4096; do
    echo "===== K=$K N=4096 ====="
    "$PYTHON" -m moe_bench.tools.verify_fp8_gemm 64 256 "$K" 4096 20 || return 1
  done > "$OUT/k_sweep_N4096.txt" 2>&1
  return 0
}

# ---------- P3: NCU 归因（4 份 rep） ----------
p3() {
  command -v ncu >/dev/null || { note "P3: 找不到 ncu，跳过"; return 1; }
  local SECS=(--section SpeedOfLight --section ComputeWorkloadAnalysis
              --section InstructionStats --section SchedulerStats
              --section WarpStateStats --section MemoryWorkloadAnalysis
              --section SourceCounters --section Occupancy --section LaunchStats)
  run_ncu() {  # $1=名字 $2=kernel正则 $3=K $4=N
    note "P3: ncu $1"
    ncu --replay-mode kernel --kernel-name-base demangled -f \
        -k "$2" --launch-skip 4 --launch-count 1 "${SECS[@]}" \
        --export "$OUT/$1" \
        "$PYTHON" -m moe_bench.tools.verify_fp8_gemm 64 256 "$3" "$4" 10 \
        > "$OUT/$1.run.log" 2>&1 || return 1
    ncu --import "$OUT/$1.ncu-rep" --page details > "$OUT/$1.details.txt" 2>&1
  }
  local rc=0
  run_ncu ncu_L0_fp8 'regex:gg8::kernel\(' 4096 1536 || rc=1
  run_ncu ncu_L0_raw 'regex:kernel_raw'    4096 1536 || rc=1
  run_ncu ncu_L1_fp8 'regex:gg8::kernel\(' 768  4096 || rc=1
  run_ncu ncu_L1_raw 'regex:kernel_raw'    768  4096 || rc=1
  return $rc
}

# ---------- P4: TASK_Q 2→4 A/B（改正本→测→还原） ----------
p4() {
  note "P4: TASK_Q 2→4（改正本 $HDR，测完自动还原）"
  grep -q 'static constexpr int TASK_Q = 2;' "$HDR" || { note "P4: 找不到 TASK_Q = 2 定义行，放弃"; return 1; }
  cp "$HDR" "$HDR.probe_bak"
  sed -i 's/static constexpr int TASK_Q = 2;/static constexpr int TASK_Q = 4;/' "$HDR"
  grep -q 'TASK_Q = 4' "$HDR" || { mv "$HDR.probe_bak" "$HDR"; return 1; }
  local rc=0
  rm -rf moe_bench/kernels/tk/build
  if "$PYTHON" moe_bench/kernels/tk/build.py 4 > "$OUT/build_log_taskq4.txt" 2>&1; then
    run_verify_set "$OUT/taskq4_verify.txt" || rc=1
    cuobjdump -res-usage "$SO" > "$OUT/taskq4_res_usage.txt" 2>&1
    grep -E 'FAIL' "$OUT/taskq4_verify.txt" && rc=1
  else
    rc=1
  fi
  # 还原源文件 + 基线 .so（build 缓存按文件名命中，必须清掉 TASK_Q4 的 .so）
  mv "$HDR.probe_bak" "$HDR"
  rm -rf moe_bench/kernels/tk/build
  "$PYTHON" moe_bench/kernels/tk/build.py 4 > "$OUT/build_log_restore.txt" 2>&1 \
      || note "P4: 警告：基线 .so 重建失败，下次运行前需手动重编"
  return $rc
}

# ---------- 执行 ----------
p0 && mark P0_build_baseline OK || mark P0_build_baseline FAIL
if [ -f "$SO" ]; then
  p1 && mark P1_tma_litmus OK || mark P1_tma_litmus FAIL
  p2 && mark P2_k_sweep OK || mark P2_k_sweep FAIL
  if [ "$DO_NCU" -eq 1 ]; then p3 && mark P3_ncu OK || mark P3_ncu FAIL
  else mark P3_ncu SKIPPED; fi
  if [ "$DO_TASKQ" -eq 1 ]; then p4 && mark P4_taskq4 OK || mark P4_taskq4 FAIL
  else mark P4_taskq4 SKIPPED; fi
else
  note "P0 编译失败，跳过所有 GPU 测试（build_log.txt 是本次最重要的产物）"
  p1 && mark P1_tma_litmus OK || mark P1_tma_litmus FAIL
  mark P2_k_sweep SKIPPED; mark P3_ncu SKIPPED; mark P4_taskq4 SKIPPED
fi

# ---------- 摘要 + 打包 ----------
{ echo "== STATUS =="; cat "$STATUS"; echo
  echo "== 关键计时行 =="
  grep -h '\[fp8 gemm\]' "$OUT"/baseline_verify.txt "$OUT"/k_sweep_N4096.txt \
       "$OUT"/taskq4_verify.txt 2>/dev/null
  echo; echo "== litmus 判定（CALL=syscall / UTMA*|UBLKCP=原生）=="
  cat "$OUT/tma_litmus_verdict.txt" 2>/dev/null
} > "$OUT/SUMMARY.txt"
find "$OUT" -type f -printf '%10s  %p\n' | sort -k2 > "$OUT/MANIFEST.txt"

if command -v zip >/dev/null; then
  PKG="${OUT}.zip"; (cd "$(dirname "$OUT")" && zip -qr "$(basename "$PKG")" "$(basename "$OUT")")
else
  PKG="${OUT}.tar.gz"; tar czf "$PKG" "$OUT"
fi
note "完成。产物包: $WS/$PKG"
cat "$OUT/SUMMARY.txt"
