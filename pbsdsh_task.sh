#!/usr/bin/env bash
# Per-node task for multi-node jobs, spawned by pbsdsh from
# submit_asyn_aa.pbs.
#
# Two hard-won facts about pbsdsh/TM on this site:
#   1. Tasks inherit ONLY the qsub -v job environment -- variables the job
#      script computes and exports at runtime (CONDA_BASE, NODES, NTOMP...)
#      do NOT cross over. Hence .pbsdsh_task_env, written by the job head
#      next to this script and sourced below.
#   2. pbsdsh exits 0 even when every task failed, so the head node counts
#      .task_<N>.ok sentinels instead of trusting the exit code.
# Also: pbsdsh's stdout relay is lossy on some PBS Pro builds, so
# everything is logged per node to pbsdsh_node<N>.log.
TASK_DIR="${0%/*}"
set -euo pipefail
source "$TASK_DIR/.pbsdsh_task_env"

# Chunk-packing guard: if the scheduler ever places several chunks of this
# job on ONE host (place=scatter should prevent it), every chunk spawns a
# task here -- with the SAME PBS_NODENUM (host ordinal on this site).
# Only the first task on each physical host may proceed; duplicates exit.
# Without this, concurrent setups raced and tripled a topol.top (2026-10-20).
HOST_LOCK="$REPO/.hostlock-$(hostname)"
if ! mkdir "$HOST_LOCK" 2>/dev/null; then
    echo "[task] duplicate task on $(hostname) (hostlock held) -- exiting"
    exit 0
fi
exec > "$TASK_DIR/pbsdsh_node${PBS_NODENUM:-x}.log" 2>&1

# TM strips the environment: replay the head node's binary AND library
# paths. Without LD_LIBRARY_PATH the gmx binary exists but exits 127
# (dynamic loader cannot find its shared libraries).
[ -n "${PATH_SAVE:-}" ] && export PATH="${PATH_SAVE}:$PATH"
[ -n "${LD_LIBRARY_PATH_SAVE:-}" ] && \
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_SAVE}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
# gmx MUST resolve inside this task. TM hands us a scrubbed environment, so
# the module system may not work here; the authoritative fallback is the
# install bin the head node resolved at submit time (GMX_BIN). And we
# HARD-FAIL loudly if gmx is still missing -- a silent `|| true` here cost
# us a full 4-node job to a bare-'gmx' FileNotFoundError (job 1562xx).
source /etc/profile.d/modules.sh 2>/dev/null || true
module load "$GMX_MODULE" 2>/dev/null || true
if ! command -v gmx >/dev/null 2>&1 && [ -n "${GMX_BIN:-}" ]; then
    export PATH="${GMX_BIN}:$PATH"
fi
command -v gmx >/dev/null 2>&1 || {
    echo "[task] FATAL: gmx not found (GMX_MODULE=$GMX_MODULE, " >&2
    echo "       GMX_BIN=${GMX_BIN:-unset}, PATH=$PATH)" >&2
    exit 3; }
echo "[task] gmx resolved: $(command -v gmx)"
gmx --version 2>&1 | grep -m1 -i 'gromacs version' || true
cd "$REPO"

GPUS=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
[ "${GPUS:-0}" -lt 1 ] && GPUS=1
GPU_LIST=$(seq -s, 0 $((GPUS - 1)))
echo "== node $(hostname)  index ${PBS_NODENUM}/${NODES}  gpus ${GPU_LIST} =="

case "$PHASE" in
    setup)  python3 launch_asyn_aa.py --replicas "$REPLICAS" --gpus "$GPU_LIST" \
                --us "$US" --setup-only --ff-dir "$FF_DIR" \
                --node-index "$PBS_NODENUM" --node-count "$NODES" ;;
    run)    python3 launch_asyn_aa.py --replicas "$REPLICAS" --gpus "$GPU_LIST" \
                --us "$US" --maxh "$MAXH" --ntomp "$NTOMP" \
                --retries "$RETRIES" --ff-dir "$FF_DIR" \
                --node-index "$PBS_NODENUM" --node-count "$NODES" ;;
    *)      echo "unknown PHASE=$PHASE" >&2; exit 2 ;;
esac

touch "$TASK_DIR/.task_${PBS_NODENUM:-x}.ok"
