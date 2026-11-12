"""Debug only: epa_dispatch.cuh + globaltimer marks (compiled with -DEPA_TRACE)."""
s = open('epa_dispatch.cuh').read()
pre = '''#ifdef EPA_TRACE
__device__ unsigned long long epa_trace[64][8];
__device__ __forceinline__ unsigned long long epa_now() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define EPA_T(slot) do { epa_trace[blockIdx.x][slot] = epa_now(); } while (0)
#else
#define EPA_T(slot) do {} while (0)
#endif
'''
def ins(anchor, text, before=True):
    global s
    assert s.count(anchor) == 1, anchor
    s = s.replace(anchor, text + anchor if before else anchor + text)
ins("namespace mooncake_epa {\n", pre)
ins("    const auto sm_id = static_cast<int>(blockIdx.x);\n", "    if (threadIdx.x == 0) EPA_T(0);\n", before=False)
ins("// Warp reduce: lane j keeps", "        if (lane_id == 0) EPA_T(2);\n")
ins("        if (sm_id == 0) {\n// The first SM is also", "        if (lane_id == 0) EPA_T(3);\n")
ins("    // Issue count sends\n", "    if (threadIdx.x == 0) EPA_T(4);\n")
ins("LOW_LATENCY_DISPATCH_RECV:\n", "    if (threadIdx.x == 0) EPA_T(5);\n", before=False)
end = "    } else {\n        mc_bar_sync(warp_group_id + 2, kNumWarpsPerGroup * 32);\n    }\n}\n"
assert s.count(end) == 1
s = s.replace(end, "    } else {\n        mc_bar_sync(warp_group_id + 2, kNumWarpsPerGroup * 32);\n    }\n    __syncthreads();\n    if (threadIdx.x == 0) EPA_T(6);\n}\n")
open('epa_dispatch_trace.cuh', 'w').write(s)
