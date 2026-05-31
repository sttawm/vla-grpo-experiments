#!/bin/bash
# Wait for B and E precomputes to finish, merge caches, launch FT-Plan v3 on pod 2.
set -e

EXPERIMENTS_DIR="$(cd "$(dirname "$0")" && pwd)"

A40_B="root@69.30.85.61"
A40_B_PORT=22008
A40_E="root@69.30.85.231"
A40_E_PORT=22066
POD2="root@154.54.102.31"
POD2_PORT=14822
SSH_KEY=~/.ssh/id_ed25519
SSH_OPTS="-o StrictHostKeyChecking=no"

CACHE_DIR="$EXPERIMENTS_DIR/cache"
mkdir -p "$CACHE_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Wait for E to finish ────────────────────────────────────────────────────
log "Waiting for E precompute to finish..."
while ! ssh $SSH_OPTS -i $SSH_KEY -p $A40_E_PORT $A40_E \
    "grep -q 'Done\.' /workspace/experiments/results/precompute_v3_E.log 2>/dev/null"; do
    PROG=$(ssh $SSH_OPTS -i $SSH_KEY -p $A40_E_PORT $A40_E \
        "tail -1 /workspace/experiments/results/precompute_v3_E.log 2>/dev/null")
    log "  E: $PROG"
    sleep 60
done
log "E precompute DONE."

# ── 2. Wait for B to finish ────────────────────────────────────────────────────
log "Waiting for B precompute to finish..."
while ! ssh $SSH_OPTS -i $SSH_KEY -p $A40_B_PORT $A40_B \
    "grep -q 'Done\.' /workspace/experiments/results/precompute_v3_B.log 2>/dev/null"; do
    PROG=$(ssh $SSH_OPTS -i $SSH_KEY -p $A40_B_PORT $A40_B \
        "tail -1 /workspace/experiments/results/precompute_v3_B.log 2>/dev/null")
    log "  B: $PROG"
    sleep 60
done
log "B precompute DONE."

# ── 3. Copy B, E, and A_current caches locally ────────────────────────────────
log "Copying B cache from A40-B..."
scp $SSH_OPTS -i $SSH_KEY -P $A40_B_PORT \
    $A40_B:/workspace/experiments/results/qwen3_libero_labels_v3_B.json \
    "$CACHE_DIR/qwen3_libero_labels_v3_B.json"

log "Copying E cache from A40-E..."
scp $SSH_OPTS -i $SSH_KEY -P $A40_E_PORT \
    $A40_E:/workspace/experiments/results/qwen3_libero_labels_v3_E.json \
    "$CACHE_DIR/qwen3_libero_labels_v3_E.json"

log "Copying A_current cache from main pod..."
scp $SSH_OPTS -i $SSH_KEY -P 25129 \
    root@185.216.23.177:/workspace/experiments/results/qwen3_libero_labels.json \
    "$CACHE_DIR/qwen3_libero_labels.json"

# ── 4. Push all caches to pod 2 ───────────────────────────────────────────────
log "Pushing caches to pod 2..."
scp $SSH_OPTS -i $SSH_KEY -P $POD2_PORT \
    "$CACHE_DIR/qwen3_libero_labels_v3_B.json" \
    "$CACHE_DIR/qwen3_libero_labels_v3_E.json" \
    "$CACHE_DIR/qwen3_libero_labels.json" \
    $POD2:/workspace/experiments/results/

# ── 5. Deploy merge script and run on pod 2 ───────────────────────────────────
log "Deploying merge script to pod 2..."
scp $SSH_OPTS -i $SSH_KEY -P $POD2_PORT \
    "$EXPERIMENTS_DIR/merge_v3_cache.py" $POD2:/workspace/experiments/merge_v3_cache.py
log "Running merge on pod 2..."
ssh $SSH_OPTS -i $SSH_KEY -p $POD2_PORT $POD2 \
    "cd /workspace/experiments && python3 merge_v3_cache.py"

# ── 6. Launch FT-Plan v3 on pod 2 ─────────────────────────────────────────────
log "Launching FT-Plan v3 training on pod 2..."
ssh $SSH_OPTS -i $SSH_KEY -p $POD2_PORT $POD2 \
    "cd /workspace/experiments && nohup python -u finetune_openvla_v3.py \
     > results/finetune_train_ftplan_v3.log 2>&1 &"
log "FT-Plan v3 launched. Log: pod2:/workspace/experiments/results/finetune_train_ftplan_v3.log"
log "All done."
