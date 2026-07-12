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
    # 环境干扰取证(docs/27: 双峰随轮次在配置间游走): 每步结束记录全卡时钟/占用
    nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,utilization.gpu,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | sed "s/^/${name},/" >> "$OUT/clocks_per_step.csv"
}

cd "$PARENT_DIR"

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

    # ---------- 3. 正确性对拍（harness 自动 verify vs reference_moe） ----------
    run_step 03_correct_ne64            600 python -m moe_bench.tools.run_tktp 64  --iters 10
    run_step 03p_correct_ne64_push 600 env TK_TP_DISPATCH=push python -m moe_bench.tools.run_tktp 64 --iters 10
    if [ "$QUICK" != "1" ]; then
        run_step 03_correct_ne128       600 python -m moe_bench.tools.run_tktp 128 --iters 10
        run_step 03_correct_ne256       900 python -m moe_bench.tools.run_tktp 256 --iters 10
        run_step 03_correct_ne64_skewed 600 python -m moe_bench.tools.run_tktp 64  --iters 10 --dist skewed
        run_step 03p_correct_ne256_push 900 env TK_TP_DISPATCH=push python -m moe_bench.tools.run_tktp 256 --iters 10
        run_step 03p_correct_skewed_push 600 env TK_TP_DISPATCH=push python -m moe_bench.tools.run_tktp 64 --iters 10 --dist skewed
    fi

    # ---------- 4. 性能：serial baseline vs tktp（同 harness 同 config） ----------
    run_step 04_bench_serial_512 600 python -m moe_bench.tools.run_tktp 64 --scheme serial \
        --no-verify --iters 50 --json "$JSONS/serial_ne64_t512.json"
    run_step 04_bench_tktp_512   600 python -m moe_bench.tools.run_tktp 64 --scheme tktp \
        --no-verify --iters 50 --json "$JSONS/tktp_ne64_t512.json"
    # push 路径已冻结(docs/25: TP 是 GEMM-bound, push 无收益且 scatter 粒度受限),
    # 保留单点 bench 作回归记录
    run_step 04p_bench_push_512 600 env TK_TP_DISPATCH=push python -m moe_bench.tools.run_tktp 64 \
        --scheme tktp --no-verify --iters 50 --json "$JSONS/tktp_push_ne64_t512.json"

    if [ "$QUICK" != "1" ]; then
        # ---------- 5. comm SM 预算 sweep（docs/25: 拐点在 24 之后, 扫到 40） ----------
        for CS in 8 16 24 32 40; do
            run_step "05_sweep_commsms_${CS}" 600 \
                env TK_COMM_SMS=$CS python -m moe_bench.tools.run_tktp 64 \
                --scheme tktp --no-verify --iters 30 \
                --json "$JSONS/tktp_commsms${CS}.json"
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
note "完成: PASS=$PASS FAIL=$FAIL"
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
