#!/bin/bash
# DistMOE + EPA pool: scenario 1 (same memory as upstream -> more checkpoints) and
# scenario 2 (small pool -> less memory, one round via the lean layout). 8 GPUs, T = 4,096.
R=$(cd "$(dirname "$0")" && pwd)   # this folder
D=$R
DUMP=${DUMP:-/tmp/distmoe_dumps}
OUT=$D/results/scenarios.jsonl
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=$CUDA_HOME/bin:$PATH
PY=${PY:-python}
TR="${TORCHRUN:-torchrun} --nproc-per-node 8"
while true; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024' | wc -l)
  [ "$busy" = 0 ] && break
  sleep 60
done
echo "$(date '+%F %T') GPUs free, starting"
exec 9>${GPU_LOCK:-/tmp/gpu.lock}; flock 9
rm -f $D/foreign_sc.txt; mkdir -p $DUMP $D/results
( while true; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
      u=$(ps -o user= -p $p 2>/dev/null)
      if [ -n "$u" ] && [ "$u" != "$(whoami)" ]; then echo "$(date +%T) $u $p" >> $D/foreign_sc.txt; fi
    done; sleep 5
  done ) & WATCH=$!
cd $D
run() { local name=$1; shift; echo "$(date +%T) start $name"; timeout 1500 $TR --master-port $((29900 + RANDOM % 90)) epa_train.py --tag $name "$@" > $DUMP/log_sc_$name.txt 2>&1; echo "$(date +%T) $name rc=$?"; grep -h "^RESULT\|Error\|error" $DUMP/log_sc_$name.txt | cut -c1-160 | head -3; }
cmp() { echo "--- $1 vs $2"; $PY compare_dumps.py $DUMP/$1 $DUMP/$2; }
# Correctness (2 layers): every EPA case against the upstream package
C="--layers 2 --iters 3 --warmup 1"
run up_uni_min $C --pkg upstream --routing uniform --slot min --dump $DUMP/up_uni_min
run up_ext_min $C --pkg upstream --routing extreme --slot min --dump $DUMP/up_ext_min
run c_up_uni_save $C --pkg upstream --routing uniform --slot 2 --dump $DUMP/c_up_uni_save
run c_s1_uni  $C --pkg fork --epa 2    --pool 11.09 --routing uniform --dump $DUMP/c_s1_uni
run c_s1_ext  $C --pkg fork --epa 2    --pool 11.09 --routing extreme --dump $DUMP/c_s1_ext
run c_s2_ext  $C --pkg fork --epa 1.25 --pool 2.0   --routing extreme --dump $DUMP/c_s2_ext
cmp up_uni_min c_up_uni_save; cmp up_uni_min c_s1_uni; cmp up_ext_min c_s1_ext; cmp up_ext_min c_s2_ext
# Timed (4 layers, 10 steps), one CUDA graph per step and eager
for mode in graph eager; do
  G=""; [ $mode = graph ] && G="--graph"
  P="--layers 4 --iters 10 --warmup 3 $G --out $OUT"
  for rt in uniform extreme; do
    run ${mode}_${rt}_up_min      $P --pkg upstream --routing $rt --slot min
    run ${mode}_${rt}_up_s1       $P --pkg upstream --routing $rt --slot 1
    run ${mode}_${rt}_up_s16      $P --pkg upstream --routing $rt --slot 1.6
    run ${mode}_${rt}_s1_p11      $P --pkg fork --epa 2 --pool 11.09 --routing $rt
    run ${mode}_${rt}_s1_p7       $P --pkg fork --epa 2 --pool 7 --routing $rt
    run ${mode}_${rt}_s2_min      $P --pkg fork --epa 1.25 --routing $rt --slot min
    run ${mode}_${rt}_s2_min_nolean $P --pkg fork --epa 1.25 --routing $rt --slot min --no-lean
    run ${mode}_${rt}_s2_p2       $P --pkg fork --epa 1.25 --pool 2.0 --routing $rt
    run ${mode}_${rt}_s2_p2_nolean $P --pkg fork --epa 1.25 --pool 2.0 --routing $rt --no-lean
    run ${mode}_${rt}_up_min2     $P --pkg upstream --routing $rt --slot min
  done
done
kill $WATCH
if [ -s $D/foreign_sc.txt ]; then echo "other users' processes seen:"; sort -u -k2,3 $D/foreign_sc.txt | head; else echo CLEAN; fi
