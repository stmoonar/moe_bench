#!/usr/bin/env bash
# =============================================================================
# TP tile-overlap 一键测试脚本：编译 -> 调度裁决 -> 正确性对拍 -> 性能 bench -> 打包 zip
#
# 用法（在任意目录执行均可）：
#   bash moe_bench/tools/run_tp_all.sh
#
# 可覆盖的环境变量：
#   PYTHON_BIN conda Python（默认 /root/miniconda3/envs/vllm-td/bin/python）
#   VENV   可选的 activate 脚本；默认 none，直接使用 PYTHON_BIN
#   CARDS  使用的 4 卡组（如 "0,1,2,3"）；不设则按 AGENTS.md 优先级自动挑空闲组
#   QUICK  =1 只跑核心步骤（编译+裁决+NE64 对拍+512token bench），跳过 sweep
#   REUSE_BENCH =1 在一次四进程/NCCL 生命周期内跑完核心性能矩阵（默认）；
#          =0 恢复逐 case 独立进程，适合定位死锁。使用 STEPS/FOCUS 时默认自动回退为 0。
#   FOCUS  =1 本轮迭代验证集（~4 分钟）：编译 + 默认路径对拍 + 主形状 bench
#          + 本轮 A/B + t1024 + 归因。serial 基线/comm sweep/NE sweep/push 回归
#          等六轮稳定项全部跳过（serial 稳定在 ±0.3%，比率用历史 serial 即可）；
#          动了调度表用 STEPS 把 02_verify 加回来。全量回归留给里程碑/报数轮。
#   STEPS  步骤名过滤正则（grep -E），只跑匹配的步骤，其余记 SKIP。
#          例：STEPS='01_|03_correct_ne64$|08_' 只编译+默认对拍+归因。
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
REUSE_BENCH="${REUSE_BENCH:-auto}"

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

# ---------- 步骤过滤（FOCUS 预设 / STEPS 正则） ----------
# FOCUS=1: 本轮迭代验证集。必跑 00/01(复位+预检+编译), 默认路径对拍,
# 主形状 bench + 本轮 A/B(04l), t1024, 归因(08)。其余稳定项 SKIP。
if [ "${FOCUS:-0}" = "1" ] && [ -z "${STEPS:-}" ]; then
    STEPS='^00_|^01_|^03_correct_ne64$|^04_bench_tktp_512$|^04l_|^06_tktp_t1024$|^08_time_stages$'
fi
if [ "$REUSE_BENCH" = "auto" ]; then
    if [ -n "${STEPS:-}" ]; then REUSE_BENCH=0; else REUSE_BENCH=1; fi
fi
note "核心性能矩阵进程复用: REUSE_BENCH=$REUSE_BENCH"

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
    # 环境干扰取证(docs/27: 双峰随轮次在配置间游走): 每步结束记录全卡时钟/占用
    nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | sed "s/^/${name},/" >> "$OUT/clocks_per_step.csv"
}

cd "$PARENT_DIR"

# ---------- 0a. 复位 TP 调优 config（docs/29: 上一轮 TUNE 产物装在 vllm configs
# 里会让 04/06/07 的 serial 悄悄变成调优口径, 与历史失去可比性; 09 会重新生成） ----------
run_step 00_reset_tuned_cfg 120 python -c "
import glob, os
import vllm.model_executor.layers.fused_moe.fused_moe as fm
d = os.path.join(os.path.dirname(fm.__file__), 'configs')
for f in glob.glob(os.path.join(d, 'E=*,N=768,device_name=*.json')):
    os.remove(f); print('removed', f)
print('reset done')
"

# ---------- 0b. CPU 预检（无 GPU 依赖：设备守卫 + 表裁决 + 数据流模拟） ----------
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
    # ---------- 2. 调度裁决：GPU builder vs host golden + TP 不变量 ----------
    run_step 02_verify_tp_schedule 900 python -m moe_bench.tools.verify_tp_schedule
    # ---------- 2f. FP8 grouped GEMM 单卡裁决(docs/37 P1: 对拍 + TFLOP/s) ----------
    run_step 02f_verify_fp8_gemm 900 python -m moe_bench.tools.verify_fp8_gemm
    run_step 02f_verify_fp8_gemm_l1 900 python -m moe_bench.tools.verify_fp8_gemm 64 256 768 4096

    # ---------- 3. 正确性对拍（harness 自动 verify vs reference_moe） ----------
    # 默认路径现在是 L0 v2 + GLU(docs/30);再单独门控 v2 无 GLU 与 v1 回归,
    # 三者分开跑,失败时可以直接定位是 dispenser 还是 GLU store 的问题。
    run_step 03_correct_ne64            600 python -m moe_bench.tools.run_tktp 64  --iters 10
    run_step 03g_correct_ne64_gluoff 600 env TK_L0_GLU=0 python -m moe_bench.tools.run_tktp 64 --iters 10
    run_step 03o_correct_ne64_l0v1   600 env TK_L0=v1 python -m moe_bench.tools.run_tktp 64 --iters 10
    run_step 03l_correct_ne64_l1v2   600 env TK_L1=v2 python -m moe_bench.tools.run_tktp 64 --iters 10
    run_step 03p_correct_ne64_push 600 env TK_TP_DISPATCH=push python -m moe_bench.tools.run_tktp 64 --iters 10
    if [ "$QUICK" != "1" ]; then
        run_step 03_correct_ne128       600 python -m moe_bench.tools.run_tktp 128 --iters 10
        run_step 03_correct_ne256       900 python -m moe_bench.tools.run_tktp 256 --iters 10
        run_step 03_correct_ne64_skewed 600 python -m moe_bench.tools.run_tktp 64  --iters 10 --dist skewed
        run_step 03p_correct_ne256_push 900 env TK_TP_DISPATCH=push python -m moe_bench.tools.run_tktp 256 --iters 10
        run_step 03p_correct_skewed_push 600 env TK_TP_DISPATCH=push python -m moe_bench.tools.run_tktp 64 --iters 10 --dist skewed
    fi

    # ---------- 4. 性能：serial baseline vs tktp（同 harness 同 config） ----------
    # 默认由 04_bench_suite_reuse 一次性执行；逐 case 模式仅作死锁排障回退。
    if [ "$REUSE_BENCH" != "1" ]; then
    run_step 04_bench_serial_512 600 python -m moe_bench.tools.run_tktp 64 --scheme serial \
        --no-verify --iters 50 --json "$JSONS/serial_ne64_t512.json"
    run_step 04_bench_tktp_512   600 python -m moe_bench.tools.run_tktp 64 --scheme tktp \
        --no-verify --iters 50 --json "$JSONS/tktp_ne64_t512.json"
    # docs/30/32 A/B: L0 GLU off / L0 v1 / L1 v1, 每项优化独立定价
    run_step 04g_bench_gluoff_512 600 env TK_L0_GLU=0 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --no-verify --iters 50 --json "$JSONS/tktp_gluoff_ne64_t512.json"
    run_step 04o_bench_l0v1_512 600 env TK_L0=v1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --no-verify --iters 50 --json "$JSONS/tktp_l0v1_ne64_t512.json"
    run_step 04l_bench_l1v2_512 600 env TK_L1=v2 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --no-verify --iters 50 --json "$JSONS/tktp_l1v2_ne64_t512.json"
    # push 路径已冻结(docs/25: TP 是 GEMM-bound, push 无收益且 scatter 粒度受限),
    # 保留单点 bench 作回归记录
    run_step 04p_bench_push_512 600 env TK_TP_DISPATCH=push python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --no-verify --iters 50 --json "$JSONS/tktp_push_ne64_t512.json"
    fi
    # ---------- 4f. FP8 serial 基线(docs/37 P0: 先定标, 再写 kernel) ----------
    run_step 03f_correct_serial_fp8 600 python -m moe_bench.tools.run_tktp 64 \
        --scheme serial --precision fp8 --iters 10
    # ---------- 4f8. FP8 tktp(docs/39 P2: fp8 AG + fp8 GEMM + GLU, L1 bf16) ----------
    run_step 03f8_correct_tktp_fp8 600 python -m moe_bench.tools.run_tktp 64 \
        --precision fp8 --iters 10
    if [ "$REUSE_BENCH" != "1" ]; then
    run_step 04f8_bench_tktp_fp8_512 600 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_ne64_t512.json"
    run_step 04f8_bench_tktp_fp8_t1024 600 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 30 --tokens 1024 \
        --json "$JSONS/tktp_fp8_t1024.json"
    # docs/42 P3 A/B: L1 fp8 关闭档(w2 反量化 bf16 + v1)
    run_step 04l8_bench_l1fp8_off_512 600 env TK_L1_FP8=0 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_l1off_ne64_t512.json"
    fi
    # ---------- 4w8. 方案A 通信 warp 化(2026-07-16: comm 角色降为 producer
    # warp 闲置 lane, GEMM 满 SM, 回收让渡税; A/B 阶梯单独定价 L0/L1) ----------
    run_step 03w8_correct_tktp_fp8_warp 600 env TK_L0_WARP=1 TK_L1_WARP=1 \
        python -m moe_bench.tools.run_tktp 64 --precision fp8 --iters 10
    if [ "$REUSE_BENCH" != "1" ]; then
    run_step 04w8_bench_l0warp_512 600 env TK_L0_WARP=1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_l0warp_ne64_t512.json"
    run_step 04w8_bench_l1warp_512 600 env TK_L1_WARP=1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_l1warp_ne64_t512.json"
    run_step 04w8_bench_bothwarp_512 600 env TK_L0_WARP=1 TK_L1_WARP=1 \
        python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_bothwarp_ne64_t512.json"
    run_step 04w8_bench_bothwarp_t1024 600 env TK_L0_WARP=1 TK_L1_WARP=1 \
        python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 30 --tokens 1024 \
        --json "$JSONS/tktp_fp8_bothwarp_t1024.json"
    fi
    # ---------- 4c8. Copy-engine A/B(docs/43: 线上字节 0 SM, 打破零和) ----------
    run_step 03c8_correct_tktp_fp8_ce 600 env TK_L0_CE=1 TK_L1_CE=1 \
        python -m moe_bench.tools.run_tktp 64 --precision fp8 --iters 10
    if [ "$REUSE_BENCH" = "1" ]; then
        run_step 04_bench_suite_reuse 3600 python -m moe_bench.tools.run_tp_bench_suite \
            --config "$MOE_DIR/configs/tp_rtx_pro5000_4gpu_fp8.yaml" \
            --manifest "$MOE_DIR/configs/runs/tp_run_20260715_124912.yaml" \
            --output-dir "$JSONS"
    else
    run_step 04c8_bench_l0ce_512 600 env TK_L0_CE=1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_l0ce_ne64_t512.json"
    run_step 04c8_bench_l1ce_512 600 env TK_L1_CE=1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_l1ce_ne64_t512.json"
    run_step 04c8_bench_bothce_512 600 env TK_L0_CE=1 TK_L1_CE=1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_bothce_ne64_t512.json"
    run_step 04c8_bench_bothce_t1024 600 env TK_L0_CE=1 TK_L1_CE=1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 30 --tokens 1024 \
        --json "$JSONS/tktp_fp8_bothce_t1024.json"
    # CE 下 comm 块只做本地 scatter, 拐点应大幅左移 —— 重扫小值
    for CS in 4 8 12 16; do
        run_step "05c8_ce_commsms_${CS}" 600 \
            env TK_L0_CE=1 TK_L1_CE=1 TK_COMM_SMS=$CS python -m moe_bench.tools.run_tktp 64 \
            --scheme tktp --precision fp8 --no-verify --iters 30 \
            --json "$JSONS/tktp_fp8_ce_commsms${CS}.json"
    done
    fi
    run_step 08c8_time_stages_fp8_ce 900 env TK_L0_CE=1 TK_L1_CE=1 \
        python -m moe_bench.tools.time_tp_stages 64 20 512 fp8
    # ---------- 5f. fp8 comm_sms 重扫(docs/40: AG 字节减半, 拐点应左移) ----------
    if [ "$REUSE_BENCH" != "1" ]; then
    for CS in 8 12 16 24; do
        run_step "05f_fp8_commsms_${CS}" 600 \
            env TK_COMM_SMS=$CS python -m moe_bench.tools.run_tktp 64 \
            --scheme tktp --precision fp8 --no-verify --iters 30 \
            --json "$JSONS/tktp_fp8_commsms${CS}.json"
    done
    fi
    # ---------- 4p1. P1: comm 块 per-lane 自由化(PK 路线重估 2026-07-16) ----------
    run_step 03p1_correct_tktp_fp8_lane 600 env TK_L0_LANE=1 \
        python -m moe_bench.tools.run_tktp 64 --precision fp8 --iters 10
    if [ "$REUSE_BENCH" != "1" ]; then
    run_step 04p1_bench_lane_512 600 env TK_L0_LANE=1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_lane_ne64_t512.json"
    run_step 04p1_bench_lane_t1024 600 env TK_L0_LANE=1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 30 --tokens 1024 \
        --json "$JSONS/tktp_fp8_lane_t1024.json"
    # per-lane 版拐点应大幅左移(24 SM 是波同步低效的补偿, 不是带宽需要);
    # 附 4 SM 波同步对照, 分离 "per-lane 收益" 与 "少 SM 本身"
    for CS in 4 8 12 16 24; do
        run_step "05p1_lane_commsms_${CS}" 600 \
            env TK_L0_LANE=1 TK_COMM_SMS=$CS python -m moe_bench.tools.run_tktp 64 \
            --scheme tktp --precision fp8 --no-verify --iters 30 \
            --json "$JSONS/tktp_fp8_lane_commsms${CS}.json"
    done
    run_step 05p1_wave_commsms_4 600 env TK_COMM_SMS=4 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 30 \
        --json "$JSONS/tktp_fp8_wave_commsms4.json"
    fi
    # ---------- 4p2. P2: L0 push 强路径(posted write 无 RTT, 收侧本地
    # scatter; 目标 comm SM 24 -> ~8, 让渡税 183 -> ~50-70) ----------
    run_step 03p2_correct_tktp_fp8_push 600 env TK_L0_PUSH=1 \
        python -m moe_bench.tools.run_tktp 64 --precision fp8 --iters 10
    if [ "$REUSE_BENCH" != "1" ]; then
    run_step 04p2_bench_push_512 600 env TK_L0_PUSH=1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_push_ne64_t512.json"
    run_step 04p2_bench_push_t1024 600 env TK_L0_PUSH=1 python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 30 --tokens 1024 \
        --json "$JSONS/tktp_fp8_push_t1024.json"
    # push 模式收侧只剩本地操作, 拐点应大幅左移 —— 主扫小 comm_sms
    for CS in 6 8 10 12 16 24; do
        run_step "05p2_push_commsms_${CS}" 600 \
            env TK_L0_PUSH=1 TK_COMM_SMS=$CS python -m moe_bench.tools.run_tktp 64 \
            --scheme tktp --precision fp8 --no-verify --iters 30 \
            --json "$JSONS/tktp_fp8_push_commsms${CS}.json"
    done
    # push SM 数 sweep(微基准: 4 SM 打满强路径; 在 comm_sms=8 档验证)
    for PS in 2 4 6; do
        run_step "05p2b_push_psms_${PS}" 600 \
            env TK_L0_PUSH=1 TK_COMM_SMS=8 TK_L0_PUSH_SMS=$PS \
            python -m moe_bench.tools.run_tktp 64 \
            --scheme tktp --precision fp8 --no-verify --iters 30 \
            --json "$JSONS/tktp_fp8_push_psms${PS}.json"
    done
    fi
    # ---------- 4p25. P2.5: warp 协作式 scatter(攻收侧 per-token 串行等待;
    # psms=2 已实测饱和, SM 尽量给 scatter) ----------
    run_step 03p25_correct_scatwarp 600 env TK_L0_PUSH=1 TK_L0_SCAT_WARP=1 TK_L0_PUSH_SMS=2 \
        python -m moe_bench.tools.run_tktp 64 --precision fp8 --iters 10
    if [ "$REUSE_BENCH" != "1" ]; then
    run_step 04p25_bench_scatwarp_512 600 env TK_L0_PUSH=1 TK_L0_SCAT_WARP=1 TK_L0_PUSH_SMS=2 \
        python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/tktp_fp8_scatwarp_ne64_t512.json"
    run_step 04p25_bench_scatwarp_t1024 600 env TK_L0_PUSH=1 TK_L0_SCAT_WARP=1 TK_L0_PUSH_SMS=2 \
        python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --precision fp8 --no-verify --iters 30 --tokens 1024 \
        --json "$JSONS/tktp_fp8_scatwarp_t1024.json"
    # warp scatter 吞吐应数倍于 lane 版 -> 拐点应真正左移, 主扫小 comm_sms
    for CS in 4 6 8 12 16 24; do
        run_step "05p25_scatwarp_commsms_${CS}" 600 \
            env TK_L0_PUSH=1 TK_L0_SCAT_WARP=1 TK_L0_PUSH_SMS=2 TK_COMM_SMS=$CS \
            python -m moe_bench.tools.run_tktp 64 \
            --scheme tktp --precision fp8 --no-verify --iters 30 \
            --json "$JSONS/tktp_fp8_scatwarp_commsms${CS}.json"
    done
    fi
    # ---------- 8f. fp8 分阶段归因 ----------
    run_step 08f_time_stages_fp8 900 python -m moe_bench.tools.time_tp_stages 64 20 512 fp8
    # P1 归因: 最优 lane comm_sms 档的 L0 暴露(跑完 05p1 后把 TK_COMM_SMS 换成拐点值)
    run_step 08p1_time_stages_lane 900 env TK_L0_LANE=1 TK_COMM_SMS=12 \
        python -m moe_bench.tools.time_tp_stages 64 20 512 fp8
    # P2 归因: push 模式拐点档(跑完 05p2 后按拐点调 TK_COMM_SMS)
    run_step 08p2_time_stages_push 900 env TK_L0_PUSH=1 TK_COMM_SMS=8 \
        python -m moe_bench.tools.time_tp_stages 64 20 512 fp8
    # P2.5 归因: warp scatter 拐点档
    run_step 08p25_time_stages_scatwarp 900 env TK_L0_PUSH=1 TK_L0_SCAT_WARP=1 \
        TK_L0_PUSH_SMS=2 TK_COMM_SMS=8 \
        python -m moe_bench.tools.time_tp_stages 64 20 512 fp8
    # ---------- 8w8. 方案A 定位(逐迭代 + 融合税三分解: 纯GEMM上限/共存税/
    # gate-straggler 税; 裁决 L0 双稳机制①发射饥饿 vs ②TMA 队列 HoL) ----------
    run_step 08w8_diag_warp 900 python -m moe_bench.tools.diag_warp 64 30 512
    if [ "$REUSE_BENCH" != "1" ]; then
    run_step 04f_bench_serial_fp8_512 600 python -m moe_bench.tools.run_tktp 64 \
        --scheme serial --precision fp8 --no-verify --iters 50 \
        --json "$JSONS/serial_fp8_ne64_t512.json"
    run_step 04f_bench_serial_fp8_t1024 600 python -m moe_bench.tools.run_tktp 64 \
        --scheme serial --precision fp8 --no-verify --iters 30 --tokens 1024 \
        --json "$JSONS/serial_fp8_t1024.json"
    fi

    if [ "$QUICK" != "1" ]; then
        # ---------- 5. comm SM 预算 sweep（docs/30: v2 下 comm 块会转岗,
        # L0 拐点可能右移, 重扫; L1 独立预算 TK_COMM_SMS_L1 首扫） ----------
        for CS in 8 16 24 32 40; do
            run_step "05_sweep_commsms_${CS}" 600 \
                env TK_COMM_SMS=$CS python -m moe_bench.tools.run_tktp 64 \
                --scheme tktp --no-verify --iters 30 \
                --json "$JSONS/tktp_commsms${CS}.json"
        done
        # docs/34: L1 GEMM 是 SM-bound, v2 的列扫聚合不需要 comm 块守就绪序
        # (v1 需要) -> v2 独有自由度是把 L1 预算压到极小, GEMM 拿满 SM,
        # 排空靠全员转岗。扫小值找 v2 拐点。
        for CS1 in 2 4 8 16; do
            run_step "05b_sweep_l1sms_${CS1}" 600 \
                env TK_COMM_SMS_L1=$CS1 python -m moe_bench.tools.run_tktp 64 \
                --scheme tktp --no-verify --iters 30 \
                --json "$JSONS/tktp_l1sms${CS1}.json"
        done
        # ---------- 6. token 数 sweep ----------
        for T in 256 1024; do
            run_step "06_serial_t${T}" 600 python -m moe_bench.tools.run_tktp 64 --scheme serial \
                --no-verify --iters 30 --tokens "$T" --json "$JSONS/serial_t${T}.json"
            run_step "06_tktp_t${T}"   600 python -m moe_bench.tools.run_tktp 64 --scheme tktp \
                --no-verify --iters 30 --tokens "$T" --json "$JSONS/tktp_t${T}.json"
        done
        # ---------- 6b. T=1024 双峰异常对照(docs/26: comm24 时 med 10020/min 4210) ----------
        run_step 06b_tktp_t1024_rep 600 \
            python -m moe_bench.tools.run_tktp 64 --scheme tktp --no-verify --iters 30 \
            --tokens 1024 --json "$JSONS/tktp_t1024_rep.json"
        run_step 06b_tktp_t1024_cs16 600 env TK_COMM_SMS=16 \
            python -m moe_bench.tools.run_tktp 64 --scheme tktp --no-verify --iters 30 \
            --tokens 1024 --json "$JSONS/tktp_t1024_cs16.json"
        run_step 06b_stages_t1024      900 python -m moe_bench.tools.time_tp_stages 64 20 1024
        run_step 06b_stages_t1024_cs16 900 env TK_COMM_SMS=16 \
            python -m moe_bench.tools.time_tp_stages 64 20 1024
        # ---------- 7. NE sweep 性能 ----------
        for NE in 128 256; do
            run_step "07_serial_ne${NE}" 600 python -m moe_bench.tools.run_tktp "$NE" --scheme serial \
                --no-verify --iters 30 --json "$JSONS/serial_ne${NE}.json"
            run_step "07_tktp_ne${NE}"   600 python -m moe_bench.tools.run_tktp "$NE" --scheme tktp \
                --no-verify --iters 30 --json "$JSONS/tktp_ne${NE}.json"
        done
        # ---------- 7b. NE=256 + ROW_BLOCK=64（docs/25: 50% padding 归零; T5 开关） ----------
        # 先单进程预编译 rb64 变体, 防 4 个 rank 并发编译同一 .so 相互踩踏
        run_step 07b_build_rb64 1800 python "$MOE_DIR/kernels/tk/build.py" 4 64
        run_step 07b_correct_ne256_rb64 900 env TK_ROW_BLOCK=64 \
            python -m moe_bench.tools.run_tktp 256 --iters 10
        run_step 07b_tktp_ne256_rb64 600 env TK_ROW_BLOCK=64 \
            python -m moe_bench.tools.run_tktp 256 --scheme tktp --no-verify --iters 30 \
            --json "$JSONS/tktp_ne256_rb64.json"
        # 重复第二次(docs/27: 上轮此档双峰, 同 session 复跑判内因/外因)
        run_step 07b_tktp_ne256_rb64_rep 600 env TK_ROW_BLOCK=64 \
            python -m moe_bench.tools.run_tktp 256 --scheme tktp --no-verify --iters 30 \
            --json "$JSONS/tktp_ne256_rb64_rep.json"
    fi

    # ---------- 8. 分阶段归因(docs/09 三件套: 各阶段 + GEMM-alone 对照) ----------
    run_step 08_time_stages 900 python -m moe_bench.tools.time_tp_stages 64 20
    run_step 08c_time_stages_l1sms4 900 env TK_COMM_SMS_L1=4 python -m moe_bench.tools.time_tp_stages 64 20
    run_step 08o_time_stages_l0v1 900 env TK_L0=v1 python -m moe_bench.tools.time_tp_stages 64 20
    run_step 08l_time_stages_l1v2 900 env TK_L1=v2 python -m moe_bench.tools.time_tp_stages 64 20
    run_step 08p_time_stages_push 900 env TK_TP_DISPATCH=push python -m moe_bench.tools.time_tp_stages 64 20
    if [ "$QUICK" != "1" ]; then
        run_step 08_time_stages_cs8 900 env TK_COMM_SMS=8 python -m moe_bench.tools.time_tp_stages 64 20
    fi
fi

# ---------- 9. TP-T3 v2: vLLM triton 无 ray 调优(可选, TUNE=1 打开, ~30min) ----------
# v1 的 benchmark_moe.py ray 在本机卡死 2.5h 被 SIGTERM(docs/28);
# v2 = tune_moe_tp_noray.py: 先 <2min smoke 自检, 注入无效立刻失败。
# 默认只调主报数形状 E=64(查表键 E=64,N=768); NE sweep 档位用 TUNE_E="64 128 256"。
if [ "${TUNE:-0}" = "1" ] && [ "${SKIP_ALL:-0}" != "1" ]; then
    run_step 09_tune_vllm_tp 10800 bash "$MOE_DIR/tools/tune_vllm_moe_tp.sh"
    # 调优后复测 E=64 的三个 token 档(config 已装入 vllm configs;
    # tktp 不走 triton 不必复测; ne128/256 查表键不同, 默认没调, 不复测)
    run_step 09_serial_tuned_512 600 python -m moe_bench.tools.run_tktp 64 --scheme serial \
        --no-verify --iters 50 --json "$JSONS/serial_tuned_ne64_t512.json"
    run_step 09_serial_tuned_t1024 600 python -m moe_bench.tools.run_tktp 64 --scheme serial \
        --no-verify --iters 30 --tokens 1024 --json "$JSONS/serial_tuned_t1024.json"
    run_step 09_serial_tuned_t256 600 python -m moe_bench.tools.run_tktp 64 --scheme serial \
        --no-verify --iters 30 --tokens 256 --json "$JSONS/serial_tuned_t256.json"
    # 裁决: 调优 config 是否真的被 serial 用上(任一 tuned 日志仍报默认 config 即失败)
    run_step 09v_tuned_applied 60 bash -c \
        "! grep -l 'Using default MoE config' \"$LOGS\"/09_serial_tuned_*.log"
fi

# ---------- 收尾：汇总 + 打包 ----------
note "完成: PASS=$PASS FAIL=$FAIL SKIP=$SKIP${STEPS:+ (过滤: $STEPS)}"
{
    echo; echo "== 关键结果速览 =="
    for f in "$LOGS"/03_*.log "$LOGS"/04_*.log "$LOGS"/05_*.log "$LOGS"/06_*.log "$LOGS"/07_*.log "$LOGS"/09_serial_tuned_*.log; do
        [ -f "$f" ] || continue
        echo "--- $(basename "$f")"
        grep -E "verify|FAIL|ok|µs|us/iter|latency|tokens/s|mean" "$f" | tail -8
    done
    for f in "$LOGS"/06b_stages*.log "$LOGS"/08*.log; do
        [ -f "$f" ] || continue
        echo "--- $(basename "$f")"
        tail -16 "$f"
    done
    if [ -f "$LOGS/09_tune_vllm_tp.log" ]; then
        echo "--- 09_tune_vllm_tp.log (调优增益表)"
        grep -E "结果\(us\)|gain|smoke|tuned|default" "$LOGS/09_tune_vllm_tp.log" | tail -40
    fi
} >> "$SUMMARY" 2>/dev/null

ZIP="$MOE_DIR/tp_test_results/tp_run_$TS.zip"
(cd "$OUT/.." && python -m zipfile -c "$ZIP" "$(basename "$OUT")")
note "产物已打包: $ZIP"
echo
echo "======================================================================"
echo "  把这个文件拷回本地即可: $ZIP"
echo "======================================================================"
