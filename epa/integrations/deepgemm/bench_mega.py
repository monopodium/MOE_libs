"""DeepGEMM Mega MoE: worst-case ring (upstream) vs EPArena-sized ring.

Runs DeepGEMM's own tests/test_mega_moe.py (correctness vs its baseline, then
performance) from the patched copy, one process per configuration:
  ring = upstream            DG_EPA_RING_LOAD unset: ring for all R*T*k routed tokens
  ring = EPA x beta          DG_EPA_RING_LOAD = beta: ring for a receiver load of beta*T*k
Routing: DeepGEMM's own (uniform), or Zipf expert skew via EPA_SKEW (0.3, 1.0).
Modes are interleaved per repetition. A run that exceeds the timeout is recorded
as a hang (the failure mode of a ring too small for the schedule).

usage: python bench_mega.py --tokens 1024,4096,8192 --reps 2 --out results/mega.json
"""
import argparse
import json
import os
import re
import statistics as st
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST = HERE / 'DeepGEMM-epa/tests/test_mega_moe.py'
PY = '/home/Shuang/envs/deepinfer/bin/python'


def run(T, skew, load, timeout):
    env = dict(os.environ, PYTHONPATH=str(HERE / 'DeepGEMM-epa'), EP_DISABLE_GIN='1',
               CUDA_HOME='/home/Shuang/envs/cuda132', CC='/usr/bin/gcc', CXX='/usr/bin/g++',
               DG_JIT_CACHE_DIR='/run/user/1008/epa_dg_cache', EPA_SKEW=str(skew), EPA_SKIP_BASELINE_PERF='1')
    env['PATH'] = env['CUDA_HOME'] + '/bin:' + env['PATH']
    env.pop('DG_EPA_RING_LOAD', None)
    if load is not None:
        env['DG_EPA_RING_LOAD'] = str(load)
    proc = subprocess.Popen([PY, str(TEST), '--num-max-tokens-per-rank', str(T), '--num-shared-experts', '0'],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd='/tmp',
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # a ring too small for the schedule deadlocks: kill the whole process group
        os.killpg(proc.pid, 9)
        proc.communicate()
        return dict(hang=True)
    m = re.search(r'EPA buffer_bytes=(\d+) ring_rows=(\d+)', out)
    us = [float(x) for x in re.findall(r'\|\s+(\d+) us, reduction', out)]
    return dict(hang=False, passed='passed' in out and 'Traceback' not in out,
                buffer_bytes=int(m.group(1)) if m else None, ring_rows=int(m.group(2)) if m else None,
                us_median=st.median(us) if us else None, us_max=max(us) if us else None, n_ranks=len(us),
                tail=None if us else out[-1500:])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', default='1024,4096,8192')
    p.add_argument('--skews', default='0,0.3,1.0')
    p.add_argument('--loads', default='none,2,1.25')
    p.add_argument('--reps', type=int, default=2)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--out', default=str(HERE / 'results/mega.json'))
    a = p.parse_args()
    loads = [None if x == 'none' else float(x) for x in a.loads.split(',')]
    results = []
    for T in [int(t) for t in a.tokens.split(',')]:
        for skew in [float(s) for s in a.skews.split(',')]:
            samples = {l: [] for l in loads}
            for rep in range(a.reps):
                for load in (loads if rep % 2 == 0 else loads[::-1]):
                    r = run(T, skew, load, a.timeout)
                    samples[load].append(r)
                    print(json.dumps(dict(T=T, skew=skew, load=load, rep=rep, **{k: v for k, v in r.items() if k != 'tail'})), flush=True)
                    if r.get('tail'):
                        print(r['tail'], flush=True)
            for load, rs in samples.items():
                ok = [r for r in rs if not r['hang'] and r['us_median']]
                results.append(dict(T=T, skew=skew, routing={0: 'uniform', 0.3: 'skew', 1.0: 'extreme'}.get(skew, str(skew)),
                                    mode='upstream' if load is None else 'epa', load=load,
                                    buffer_bytes=ok[0]['buffer_bytes'] if ok else None,
                                    ring_rows=ok[0]['ring_rows'] if ok else None,
                                    us=st.median(r['us_median'] for r in ok) if ok else None,
                                    hangs=sum(r['hang'] for r in rs), passed=all(r.get('passed') for r in ok) and bool(ok)))
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(results, indent=1))
