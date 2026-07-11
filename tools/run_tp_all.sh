#!/usr/bin/env bash
# =============================================================================
# TP tile-overlap 一键测试脚本：编译 -> 调度裁决 -> 正确性对拍 -> 性能 bench -> 打包 zip
#
# 用法（在任意目录执行均可）：
#   bash moe_bench/tools/run_tp_all.sh
#
# 可覆盖的环境变量：
#   VENV   python 虚拟环境 activate 脚本（默认 /data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate；
#          设为 "none" 表示当前环境已就绪，跳过 source）
#   CARDS  使用的 4 卡组（如 "9,11,13,15"）；不设则按 AGENTS.md 优先级自动挑空闲组
#   QUICK  =1 只跑核心步骤（编译+裁决+NE64 对拍+512token bench），跳过 sweep
#
# 产物：moe_bench/tp_test_results/tp_run_<时间戳>.zip（logs/ + json/ + summary.txt）
# 每一步单独 timeout，单步失败不中断后续步骤（失败记录在 summary.txt）。
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOE_DIR="$(dirname "$SCRIPT_DIR")"                 # .../moe_bench
PARENT_DIR="$(dirname "$MOE_DIR")"                 # 必须从这里跑 python -m moe_bench.*
VENV="${VENV:-/data/cinnzhang_vllm_td_test/venvs/vllm-td/bin/activate}"
QUICK="${QUICK:-0}"

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
    nvcc --version 2>/dev/null | tail -1
} > "$LOGS/00_env.log" 2>&1
note "环境信息 -> logs/00_env.log"

# ---------- 挑卡 ----------
if [ -z "${CARDS:-}" ]; then
    CARDS="$(python - <<'EOF'
import subprocess
out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
busy = set()
for line in out.strip().splitlines():
    idx, mem, util = [int(x.strip()) for x in line.split(",")]
    if mem > 2000 or util > 10:
        busy.add(idx)
for grp in ("9,11,13,15", "8,10,12,14", "1,3,5,7", "0,2,4,6"):
    ids = [int(x) for x in grp.split(",")]
    if all(i not in busy for i in ids):
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
PASS=0; FAIL=0
run_step() {  # run_step <名字> <超时秒> <命令...>
    local name="$1" tmo="$2"; shift 2
    local log="$LOGS/${name}.log"
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
}

cd "$PARENT_DIR"

# ---------- 1. 干净编译（清缓存防 .so 过期） ----------
rm -rf "$MOE_DIR/kernels/tk/build"
run_step 01_build 1800 python "$MOE_DIR/kernels/tk/build.py" 4

# 编译失败则后续全部无意义，直接打包退出
if [ ! -f "$MOE_DIR/kernels/tk/build/"tk_moe_w4_h4096_rb128.so ]; then
    note "编译产物缺失，跳过全部运行步骤"
else
    # ---------- 2. 调度裁决：GPU builder vs host golden + TP 不变量 ----------
    run_step 02_verify_tp_schedule 900 python -m moe_bench.tools.verify_tp_schedule

    # ---------- 3. 正确性对拍（harness 自动 verify vs reference_moe） ----------
    run_step 03_correct_ne64            600 python -m moe_bench.tools.run_tktp 64  --iters 10
    if [ "$QUICK" != "1" ]; then
        run_step 03_correct_ne128       600 python -m moe_bench.tools.run_tktp 128 --iters 10
        run_step 03_correct_ne256       900 python -m moe_bench.tools.run_tktp 256 --iters 10
        run_step 03_correct_ne64_skewed 600 python -m moe_bench.tools.run_tktp 64  --iters 10 --dist skewed
    fi

    # ---------- 4. 性能：serial baseline vs tktp（同 harness 同 config） ----------
    run_step 04_bench_serial_512 600 python -m moe_bench.tools.run_tktp 64 --scheme serial \
        --no-verify --iters 50 --json "$JSONS/serial_ne64_t512.json"
    run_step 04_bench_tktp_512   600 python -m moe_bench.tools.run_tktp 64 --scheme tktp \
        --no-verify --iters 50 --json "$JSONS/tktp_ne64_t512.json"

    if [ "$QUICK" != "1" ]; then
        # ---------- 5. comm SM 预算 sweep（experience/12：PCIe 上 2~8 起步） ----------
        for CS in 4 8 16 24; do
            TK_COMM_SMS=$CS run_step "05_sweep_commsms_${CS}" 600 \
                python -m moe_bench.tools.run_tktp 64 --scheme tktp --no-verify --iters 30 \
                --json "$JSONS/tktp_commsms${CS}.json"
        done
        # ---------- 6. token 数 sweep ----------
        for T in 256 1024; do
            run_step "06_serial_t${T}" 600 python -m moe_bench.tools.run_tktp 64 --scheme serial \
                --no-verify --iters 30 --tokens "$T" --json "$JSONS/serial_t${T}.json"
            run_step "06_tktp_t${T}"   600 python -m moe_bench.tools.run_tktp 64 --scheme tktp \
                --no-verify --iters 30 --tokens "$T" --json "$JSONS/tktp_t${T}.json"
        done
        # ---------- 7. NE sweep 性能 ----------
        for NE in 128 256; do
            run_step "07_serial_ne${NE}" 600 python -m moe_bench.tools.run_tktp "$NE" --scheme serial \
                --no-verify --iters 30 --json "$JSONS/serial_ne${NE}.json"
            run_step "07_tktp_ne${NE}"   600 python -m moe_bench.tools.run_tktp "$NE" --scheme tktp \
                --no-verify --iters 30 --json "$JSONS/tktp_ne${NE}.json"
        done
    fi
fi

# ---------- 收尾：汇总 + 打包 ----------
note "完成: PASS=$PASS FAIL=$FAIL"
{
    echo; echo "== 关键结果速览 =="
    for f in "$LOGS"/03_*.log "$LOGS"/04_*.log "$LOGS"/05_*.log "$LOGS"/06_*.log "$LOGS"/07_*.log; do
        [ -f "$f" ] || continue
        echo "--- $(basename "$f")"
        grep -E "verify|FAIL|ok|µs|us/iter|latency|tokens/s|mean" "$f" | tail -8
    done
} >> "$SUMMARY" 2>/dev/null

ZIP="$MOE_DIR/tp_test_results/tp_run_$TS.zip"
(cd "$OUT/.." && python -m zipfile -c "$ZIP" "$(basename "$OUT")")
note "产物已打包: $ZIP"
echo
echo "======================================================================"
echo "  把这个文件拷回本地即可: $ZIP"
echo "======================================================================"
