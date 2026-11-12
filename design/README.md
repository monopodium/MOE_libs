# design/

| File | What |
|---|---|
| `EPArena.md` | Design of EPArena, the on-device VMM-free allocator for EP buffers, with the measured summary |
| `results.md` | Every benchmark run (generated) |
| `fig_memory.png`, `fig_time.png` | Memory and step-time figures (generated) |
| `patches/` | The integrations as diffs against upstream Mooncake EP and DeepEP V2.5, and `mooncake_epa_pull.cuh` (EPA-pull kernels, the recommended design) |
| `make_results.py` | Regenerates `results.md` and the figures from `../epa/**/results/*.json` |

Code lives in `../epa/`:

| Path | What |
|---|---|
| `include/epa/epa.cuh` | Device API: `PeerTable`, `TieredSlots`, `RemotePool`, `Stats` |
| `include/epa/heap.hpp` | Host: `SymmetricHeap` (cudaMalloc + cudaIpc), `plan_regions` (phase planner) |
| `bench/micro.py`, `csrc/micro.cu` | Primitive costs (NVLink fetch-add, push vs pull) |
| `integrations/mooncake/` | `epa_pull.cuh` (EPA-pull dispatch/combine), `make_epa_kernels.py` (generates the EPA kernels from upstream headers), `harness.cu` (NVLink harness, no TransferEngine), `bench_mc.py` |
| `integrations/deepep/` | `DeepEP-epa/` (patched copy, git history = diff), `bench_deepep.py` |

Reproduce (8 GPUs, `deepinfer` env):

```bash
python ../epa/bench/micro.py
python ../epa/integrations/mooncake/bench_mc.py --tokens 128,256,512,1024,2048,4096 \
    --routing uniform,skew,extreme --tiers 0:0,0:1.25,0:2,2:0.5 --full-cap 0 --fp8 0,1 \
    --out ../epa/integrations/mooncake/results/mc_sweep.json
python ../epa/integrations/mooncake/bench_mc.py --tokens 256,1024,4096 --tiers 1:2,2:2 --full-cap 0 \
    --out ../epa/integrations/mooncake/results/mc_recommended.json
python ../epa/integrations/deepep/bench_deepep.py --tokens 1024,4096,8192 --fp8 0,1 --reps 6 \
    --tiers 0:0,0:1.25,0.5:0.75,0.75:0.5 --out ../epa/integrations/deepep/results/deepep_sweep.json
python ../epa/integrations/mooncake/bench_mc.py --tokens 128,256,512,1024,2048,4096 --tiers 2:2 --full-cap 0 \
    --pull 1 --fp8 0,1 --out ../epa/integrations/mooncake/results/mc_pull.json      # add --pull-blocks 48 for upstream's SM count
uv run --no-project --with matplotlib make_results.py
```
DeepEP-epa must be built first (`python setup.py build_ext --inplace` with the CUDA 13.2
toolkit, see the DeepEP README); DeepEP's own tests pass with `EP_EPA_CAP`/`EP_EPA_POOL` set.
