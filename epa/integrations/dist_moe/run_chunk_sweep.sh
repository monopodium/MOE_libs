#!/usr/bin/env bash
# dist_moe EPA chunked scratch: F = 8 kept (any routing fits), scratch for chunk x balanced rows.
# CUDA-graph timing; modes interleaved per repetition. One torchrun per run.
cd /tmp
export CUDA_HOME=/home/Shuang/envs/cuda132 PATH=/home/Shuang/envs/cuda132/bin:$PATH BENCH_GRAPH=1
export PYTHONPATH=/home/Shuang/shuangma/mkernel_project/MOE_libs/epa/integrations/dist_moe/dist_moe-epa
HERE=/home/Shuang/shuangma/mkernel_project/MOE_libs/epa/integrations/dist_moe
OUT=$HERE/results/chunk_sweep.jsonl
: > $OUT
for T in 1024 4096; do for R in uniform skew extreme; do for rep in 0 1; do
  modes="none 2 1"; [ $rep = 1 ] && modes="1 2 none"
  for m in $modes; do
    if [ $m = none ]; then unset DIST_MOE_EPA_CHUNK; else export DIST_MOE_EPA_CHUNK=$m; fi
    line=$(timeout 900 /ssd1/model/shuangma/envs/distmoe/bin/torchrun --standalone --nproc-per-node=8 \
      $HERE/bench_distmoe.py $T 8 $R 20 2>&1 | tee /tmp/distmoe_last.log | grep "^RESULT" | sed 's/^RESULT //')
    [ -z "$line" ] && line="{\"T\": $T, \"F\": 8, \"routing\": \"$R\", \"epa_chunk\": \"$m\", \"failed\": true}"
    echo "$line" | sed "s/^{/{\"rep\": $rep, /" | tee -a $OUT
  done
done; done; done
