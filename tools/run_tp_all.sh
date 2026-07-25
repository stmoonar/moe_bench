#!/usr/bin/env bash
# =============================================================================
# TP 通算融合一键回归：CPU 预检 -> 编译 -> GEMM 裁决 -> 正确性对拍
#                      -> 性能(tktp vs serial) -> 分阶段归因 -> 打包 zip
#
# 用法（在任意目录执行均可）：
#   bash moe_bench/tools/run_tp_all.sh
#
# 可覆盖的环境变量：
#   PYTHON_BIN conda Python（默认 /root/miniconda3/envs/vllm-td/bin/python）
#   VENV   可选的 activate 脚本；默认 none，直接使用 PYTHON_BIN
#   CARDS  使用的 4 卡组（如 "0,1,2,3"）；不设则按 AGENTS.md 优先级自动挑空闲组
#   QUICK  =1 只跑核心步骤（预检+编译+对拍+主形状 bench+归因），跳过 sweep
#   STEPS  步骤名过滤正则（grep -E），只跑匹配的步骤，其余记 SKIP。
#          例：STEPS='01_|03_|04_' 只编译+对拍+主形状 bench。
#
# 口径：所有 benchmark 步骤都走 tools/run_tktp.py，它直接读主配置
#       configs/tp_rtx_pro5000_4gpu_fp8.yaml，命令行只做最小覆盖并打印。
#
# 产物：moe_bench/tp_test_results/tp_run_<时间戳>.zip（logs/ + json/ + summary.txt）
# 每一步单独 timeout，单步失败不中断后续步骤（失败记录在 summary.txt）。
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOE_DIR="$(dirname "$SCRIPT_DIR")"                 # .../moe_bench
PARENT_DIR="$(dirname "$MOE_DIR")"                 # 必须从这里跑 python -m moe_bench.*
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/vllm-td/bin/python}"
VENV="${VENV:-none}"
QUICK="${QUICK:-0}"
CFG="$MOE_DIR/configs/tp_rtx_pro5000_4gpu_fp8.yaml"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$MOE_DIR/tp_test_results/tp_run_$TS"
LOGS="$OUT/logs"
JSONS="$OUT/json"
mkdir -p "$LOGS" "$JSONS"
SUMMARY="$OUT/summary.txt"
touch "$SUMMARY"

note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

# ---------- 前置检查 ----------
if [ "$(basename "$MOE_DIR")" != "moe_bench" ]; then
    note "FATAL: 目录名必须是 moe_bench（当前 $(basename "$MOE_DIR")），否则 python -m moe_bench.* 找不到模块"
    exit 1
fi
if [ ! -f "$MOE_DIR/ThunderKittens/include/kittens.cuh" ]; then
    note "FATAL: 找不到 $MOE_DIR/ThunderKittens/include/kittens.cuh（TK submodule 在 moe_bench 内）。"
    note "       submodule 未初始化的话先执行: git -C $MOE_DIR submodule update --init"
    exit 1
fi
if [ "$VENV" != "none" ]; then
    if [ -f "$VENV" ]; then
        # shellcheck disable=SC1090
        source "$VENV"
        note "venv: $VENV"
    else
        note "WARN: venv 不存在（$VENV），尝试用当前环境继续。可用 VENV=none 静默。"
    fi
elif [ -x "$PYTHON_BIN" ]; then
    export PATH="$(dirname "$PYTHON_BIN"):$PATH"
    note "python: $PYTHON_BIN"
else
    note "FATAL: PYTHON_BIN 不可执行: $PYTHON_BIN"
    exit 1
fi
command -v python >/dev/null || { note "FATAL: 没有 python"; exit 1; }
command -v nvcc  >/dev/null || note "WARN: PATH 里没有 nvcc，编译步骤可能失败"

# ---------- 环境信息 ----------
{
    echo "== host ==";       hostname; date
    echo "== git ==";        git -C "$MOE_DIR" rev-parse HEAD 2>/dev/null; git -C "$MOE_DIR" branch --show-current 2>/dev/null
    echo "== nvidia-smi =="; nvidia-smi
    echo "== topo ==";       nvidia-smi topo -m
    echo "== versions ==";   python -c "import torch,sys; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda)"
    echo "== config ==";     cat "$CFG"
    nvcc --version 2>/dev/null | tail -1
} > "$LOGS/00_env.log" 2>&1
note "环境信息 -> logs/00_env.log"

# ---------- 挑卡 ----------
if [ -z "${CARDS:-}" ]; then
    CARDS="$(python - <<'EOF'
import subprocess
out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
available, busy = set(), set()
for line in out.strip().splitlines():
    idx, mem, util = [int(x.strip()) for x in line.split(",")]
    available.add(idx)
    if mem > 2000 or util > 10:
        busy.add(idx)
for grp in ("0,1,2,3", "4,5,6,7"):
    ids = [int(x) for x in grp.split(",")]
    if all(i in available and i not in busy for i in ids):
        print(grp)
        break
else:
    print("")
EOF
)"
fi
if [ -z "$CARDS" ]; then
    note "FATAL: 没有找到空闲的 4 卡组（或机器卡数不足），请人工用 CARDS=a,b,c,d 指定"
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$CARDS"
note "使用 GPU 组: $CARDS (CUDA_VISIBLE_DEVICES)"

# ---------- 通用步骤执行器 ----------
PASS=0; FAIL=0; SKIP=0
run_step() {  # run_step <名字> <超时秒> <命令...>
    local name="$1" tmo="$2"; shift 2
    local log="$LOGS/${name}.log"
    if [ -n "${STEPS:-}" ] && ! echo "$name" | grep -qE "$STEPS"; then
        SKIP=$((SKIP+1))
        return 0
    fi
    note "STEP $name: $*"
    if timeout --kill-after=30 "$tmo" "$@" > "$log" 2>&1; then
        note "  -> OK"
        PASS=$((PASS+1))
    else
        local rc=$?
        note "  -> FAIL (rc=$rc$( [ $rc -eq 124 ] && echo ', TIMEOUT'))  详见 logs/${name}.log"
        tail -20 "$log" | sed 's/^/      /' >> "$SUMMARY"
        FAIL=$((FAIL+1))
    fi
    # 环境干扰取证（docs/04：共享机器上的双峰会随轮次在配置间游走）
    nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | sed "s/^/${name},/" >> "$OUT/clocks_per_step.csv"
}

cd "$PARENT_DIR"

# ---------- 0a. 清掉 vllm 里可能残留的调优 config ----------
# 否则 serial 基线会悄悄变成"被调优过"的口径，与历史结果失去可比性。
run_step 00_reset_tuned_cfg 120 python -c "
import glob, os
import vllm.model_executor.layers.fused_moe.fused_moe as fm
d = os.path.join(os.path.dirname(fm.__file__), 'configs')
for f in glob.glob(os.path.join(d, 'E=*,N=768,device_name=*.json')):
    os.remove(f); print('removed', f)
print('reset done')
"

# ---------- 0b. CPU 预检（无 GPU 依赖：设备守卫 + 调度表裁决 + 数据流模拟） ----------
run_step 00_preflight_cpu 600 python "$MOE_DIR/tools/preflight_tp_cpu.py"
if [ "$FAIL" -gt 0 ]; then
    note "CPU 预检失败, 不烧卡, 直接打包退出"
    SKIP_ALL=1
fi

# ---------- 1. 干净编译（清缓存防 .so 过期） ----------
if [ "${SKIP_ALL:-0}" != "1" ]; then
rm -rf "$MOE_DIR/kernels/tk/build"
run_step 01_build 1800 python "$MOE_DIR/kernels/tk/build.py" 4
fi

# 预检/编译失败则后续全部无意义，直接打包退出
if [ "${SKIP_ALL:-0}" = "1" ] || [ ! -f "$MOE_DIR/kernels/tk/build/"tk_moe_w4_h4096_rb128.so ]; then
    note "预检失败或编译产物缺失，跳过全部运行步骤"
else
    # ---------- 2. FP8 grouped GEMM 单卡裁决（对拍 + 重标定代价） ----------
    run_step 02_verify_fp8_gemm_l0 900 python -m moe_bench.tools.verify_fp8_gemm
    run_step 02_verify_fp8_gemm_l1 900 python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096

    # ---------- 3. 全链路正确性（harness 自动 verify vs reference_moe） ----------
    run_step 03_correct_tktp    600 python -m moe_bench.tools.run_tktp --iters 10
    run_step 03_correct_serial  600 python -m moe_bench.tools.run_tktp --scheme serial --iters 10
    if [ "$QUICK" != "1" ]; then
        run_step 03_correct_ne128   600 python -m moe_bench.tools.run_tktp 128 --iters 10
        run_step 03_correct_ne256   900 python -m moe_bench.tools.run_tktp 256 --iters 10
        run_step 03_correct_skewed  600 python -m moe_bench.tools.run_tktp --iters 10 --dist skewed
    fi

    # ---------- 4. 性能：主配置口径 T=512，以及 T=1024 对照 ----------
    run_step 04_bench_tktp_512 600 python -m moe_bench.tools.run_tktp \
        --no-verify --json "$JSONS/tktp_t512.json"
    run_step 04_bench_serial_512 600 python -m moe_bench.tools.run_tktp --scheme serial \
        --no-verify --json "$JSONS/serial_t512.json"
    run_step 04_bench_tktp_t1024 600 python -m moe_bench.tools.run_tktp \
        --no-verify --iters 30 --tokens 1024 --json "$JSONS/tktp_t1024.json"
    run_step 04_bench_serial_t1024 600 python -m moe_bench.tools.run_tktp --scheme serial \
        --no-verify --iters 30 --tokens 1024 --json "$JSONS/serial_t1024.json"

    # ---------- 5. 分阶段归因（L0/L1 暴露 = 融合 - 纯算） ----------
    run_step 05_time_stages 900 python -m moe_bench.tools.time_tp_stages 64 20 512

    if [ "$QUICK" != "1" ]; then
        # ---------- 6. comm SM 预算 sweep（换机器/换拓扑后必须重扫拐点） ----------
        for CS in 8 12 16 24 32; do
            run_step "06_sweep_commsms_${CS}" 600 \
                env TK_COMM_SMS=$CS python -m moe_bench.tools.run_tktp \
                --no-verify --iters 30 --json "$JSONS/tktp_commsms${CS}.json"
        done
        # push 块数 sweep（其余 comm 块做本地 scatter）
        for PS in 2 4 6; do
            run_step "06_sweep_pushsms_${PS}" 600 \
                env TK_L0_PUSH_SMS=$PS python -m moe_bench.tools.run_tktp \
                --no-verify --iters 30 --json "$JSONS/tktp_pushsms${PS}.json"
        done
        # ---------- 7. NE sweep（padding 粒度敏感度，docs/09） ----------
        for NE in 128 256; do
            run_step "07_tktp_ne${NE}"   600 python -m moe_bench.tools.run_tktp "$NE" \
                --no-verify --iters 30 --json "$JSONS/tktp_ne${NE}.json"
            run_step "07_serial_ne${NE}" 600 python -m moe_bench.tools.run_tktp "$NE" \
                --scheme serial --no-verify --iters 30 --json "$JSONS/serial_ne${NE}.json"
        done
    fi
fi

# ---------- 收尾：汇总 + 打包 ----------
note "完成: PASS=$PASS FAIL=$FAIL SKIP=$SKIP${STEPS:+ (过滤: $STEPS)}"
{
    echo; echo "== 关键结果速览 =="
    for f in "$LOGS"/02_*.log "$LOGS"/03_*.log "$LOGS"/04_*.log "$LOGS"/06_*.log "$LOGS"/07_*.log; do
        [ -f "$f" ] || continue
        echo "--- $(basename "$f")"
        grep -E "verify|FAIL|ok|µs|us/iter|latency|tokens/s|mean|rel_err" "$f" | tail -8
    done
    for f in "$LOGS"/05_*.log; do
        [ -f "$f" ] || continue
        echo "--- $(basename "$f")"
        tail -16 "$f"
    done
} >> "$SUMMARY" 2>/dev/null

ZIP="$MOE_DIR/tp_test_results/tp_run_$TS.zip"
(cd "$OUT/.." && python -m zipfile -c "$ZIP" "$(basename "$OUT")")
note "产物已打包: $ZIP"
echo
echo "======================================================================"
echo "  把这个文件拷回本地即可: $ZIP"
echo "======================================================================"
