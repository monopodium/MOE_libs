#!/usr/bin/env bash
# dist_moe capacity-factor sweep: one torchrun per (T, routing, F); a trap is recorded as such.
cd /tmp
export CUDA_HOME=/home/Shuang/envs/cuda132 PATH=/home/Shuang/envs/cuda132/bin:$PATH
OUT=/home/Shuang/shuangma/mkernel_project/MOE_libs/epa/integrations/dist_moe/results/sweep.jsonl
: > $OUT
for T in 1024 4096; do for R in uniform skew extreme; do for F in 8 2 1.25 1; do
  line=$(timeout 900 /ssd1/model/shuangma/envs/distmoe/bin/torchrun --standalone --nproc-per-node=8 \
    /home/Shuang/shuangma/mkernel_project/MOE_libs/epa/integrations/dist_moe/bench_distmoe.py $T $F $R 20 2>&1 | tee /tmp/distmoe_last.log | grep "^RESULT" | sed 's/^RESULT //')
  if [ -z "$line" ]; then
    why=$(grep -m1 -oE "device-side assert|trap|illegal|capacity|Timeout|timed out|Error[^\"]{0,80}" /tmp/distmoe_last.log | head -1)
    line="{\"T\": $T, \"F\": $F, \"routing\": \"$R\", \"failed\": true, \"why\": \"$why\"}"
  fi
  echo "$line" | tee -a $OUT
done; done; done
