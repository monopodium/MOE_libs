"""Build design/results.md and its figures from the benchmark JSON.

usage: uv run --no-project --with matplotlib make_results.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(__file__).parent
EPA = D.parent / 'epa'
MCR = EPA / 'integrations/mooncake/results'
MC = [MCR / 'mc_sweep.json', MCR / 'mc_recommended.json']
MC_PULL = [MCR / 'mc_pull.json', MCR / 'mc_pull48.json']
DE = EPA / 'integrations/deepep/results/deepep_sweep.json'
MICRO = EPA / 'bench/results/micro.json'
GiB = 2 ** 30


def load(paths):
    out = []
    for p in paths if isinstance(paths, list) else [paths]:
        if p.exists():
            rows = json.loads(p.read_text())
            if p.name == 'mc_pull48.json':
                rows = [r for r in rows if r['mode'] == 'pull']
                for r in rows:
                    r['mode'] = 'pull (48 CTAs)'
            out += rows
    return out


def label(r, size_key):
    if r['mode'] == 'upstream':
        return 'upstream'
    if r['mode'].startswith('pull'):
        return 'EPA-pull' + r['mode'][4:]
    return f"EPA cap={r['cap']} pool={r['pool']} ({cfg_name(r)})"


def rel_table(rows, size_key, time_key, title):
    """Markdown table: one row per (T, routing, dtype, mode); time relative to upstream."""
    up = {(r['T'], r['routing'], r['fp8']): r for r in rows if r['mode'] == 'upstream'}
    lines = [f'#### {title}', '',
             '| T | routing | dtype | mode | memory | dispatch µs | combine µs | step µs | step vs upstream | tier 2 / tier 3 | bit-equal |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        u = up.get((r['T'], r['routing'], r['fp8']))
        rel = f"{(r[time_key] / u[time_key] - 1) * 100:+.0f}%" if u and r is not u else ('—' if r is u else 'n/a')
        mem = r[size_key] / GiB
        mem_s = f'{mem:.3f} GiB' if mem < 10 else f'{mem:.1f} GiB'
        if u and r is not u:
            mem_s += f' ({u[size_key] / r[size_key]:.0f}× less)' if u[size_key] > r[size_key] * 1.05 else ''
        t23 = (f"{r.get('tier2_frac', 0) * 100:.0f}% / {r.get('tier3_frac', 0) * 100:.0f}%" if 'tier2_frac' in r
               else f"{r.get('beyond_cap_frac', 0) * 100:.0f}% beyond cap")
        eq = r.get('all_ranks_equal')
        lines.append(f"| {r['T']} | {r['routing']} | {'FP8' if r['fp8'] else 'BF16'} | {label(r, size_key)} | {mem_s} | "
                     f"{r['dispatch_us']:.0f} | {r['combine_us']:.0f} | {r[time_key]:.0f} | {rel} | {t23} | "
                     f"{'yes' if eq else 'NO' if eq is not None else '—'} |")
    return '\n'.join(lines) + '\n'


MC_SHOW = {'cap 2x, pool 2x': 'EPA tiered (cap 2×, pool 2×)', 'EPA-pull': 'EPA-pull (all SMs)',
           'EPA-pull (48 CTAs)': 'EPA-pull (48 CTAs, as upstream)'}
DE_SHOW = {'cap 0.5x, pool 0.75x': 'EPA recommended (cap 0.5×, pool 0.75×)', 'cap 0x, pool 0x': 'EPA pure pull (cap 0, pool 0)'}


def fig_memory(mc, de):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, rows, key, show, title, ylab in (
            (axs[0], mc, 'heap_bytes', MC_SHOW, 'Mooncake legacy EP (BF16): heap per rank', 'symmetric heap per rank (GiB)'),
            (axs[1], de, 'buffer_bytes', DE_SHOW, 'DeepEP V2.5 direct: EP buffer per rank', 'EP buffer per rank (GiB)')):
        series = defaultdict(dict)
        for r in rows:
            if r['routing'] != 'uniform' or (rows is mc and r['fp8']):
                continue
            dt = ' FP8' if r['fp8'] else (' BF16' if rows is de else '')
            if r['mode'] == 'upstream':
                series['upstream' + dt][r['T']] = r[key] / GiB
            elif cfg_name(r) in show:
                series[show[cfg_name(r)] + dt][r['T']] = r[key] / GiB
        for name, pts in sorted(series.items()):
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], 'o-' if name.startswith('upstream') else 's--', label=name,
                    lw=2.5 if name.startswith('upstream') else 1.5)
            if rows is mc and name.startswith('EPA rec'):
                for x in xs:
                    u = series['upstream'].get(x)
                    if u:
                        ax.annotate(f'{u / pts[x]:.0f}× less', (x, pts[x]), textcoords='offset points', xytext=(0, -14),
                                    ha='center', fontsize=8)
        ax.set_xscale('log', base=2)
        if rows is mc:
            ax.set_yscale('log')
        ax.set_xlabel('tokens per rank T'), ax.set_ylabel(ylab)
        ax.set_title(title, fontweight='bold')
        ax.grid(alpha=0.3, which='both'), ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(D / 'fig_memory.png', dpi=130)


def cfg_name(r):
    """Tier sizes in units of the mean load: cap per (key), pool per receiver."""
    if r['mode'].startswith('pull'):
        return 'EPA-pull' + r['mode'][4:]
    if 'mean_pair' in r:     # Mooncake: key = (expert, source); receiver load = T*k
        return f"cap {r['cap'] / r['mean_pair']:.3g}x, pool {r['pool'] / (r['T'] * 6):.3g}x"
    mp = r['T'] * 0.5506     # DeepEP: key = source rank; mean pair load = T * P(token hits a rank)
    return f"cap {r['cap'] / mp:.2g}x, pool {r['pool'] / (8 * mp):.2g}x"


def fig_time(mc, de):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'uniform': '#4C72B0', 'skew': '#DD8452', 'extreme': '#C44E52'}
    styles = [':', '--', '-']
    for ax, rows, tk, title in ((axs[0], mc, 'pair_us', 'Mooncake legacy EP, BF16: step vs upstream'),
                                (axs[1], de, 'step_us', 'DeepEP V2.5 direct, BF16: step vs upstream')):
        rows = [r for r in rows if not r['fp8']]
        up = {(r['T'], r['routing']): r[tk] for r in rows if r['mode'] == 'upstream'}
        series = defaultdict(dict)
        show = MC_SHOW if rows and 'mean_pair' in rows[0] else DE_SHOW
        for r in rows:
            if r['mode'] != 'upstream' and (r['T'], r['routing']) in up and cfg_name(r) in show:
                series[(r['routing'], show[cfg_name(r)])][r['T']] = (r[tk] / up[(r['T'], r['routing'])] - 1) * 100
        cfgs = sorted({c for _, c in series})
        for (routing, cfg), pts in sorted(series.items()):
            xs = sorted(pts)
            if len(xs) < 2:
                continue
            ax.plot(xs, [pts[x] for x in xs], color=colors[routing], ls=styles[cfgs.index(cfg) % len(styles)],
                    marker='o', ms=3, label=f'{routing}: {cfg}')
        ax.axhline(0, color='k', lw=0.8)
        ax.axhspan(-10, 10, color='0.9', zorder=0)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('tokens per rank T'), ax.set_ylabel('step time vs upstream (%)  [grey: ±10% noise band]')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.grid(alpha=0.3), ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(D / 'fig_time.png', dpi=130)


def main():
    mc, de, mcp = load(MC), load(DE), load(MC_PULL)
    micro = json.loads(MICRO.read_text()) if MICRO.exists() else {}
    fig_memory(mcp or mc, de)
    fig_time(mcp or mc, de)
    out = ['# EPArena results', '',
           'Generated by `make_results.py` from the JSON next to each benchmark. 8× B300 (NVLink, NV18), '
           'R = 8, E = 384, k = 6, H = 7168. Times are medians over ranks of the median over repetitions; '
           '"step" is dispatch + combine measured end to end (CUDA graph replay for Mooncake, CUDA events for DeepEP), '
           'which includes waiting for the slowest rank. Run-to-run noise of the step time is about ±5–10 % '
           '(larger at T = 128).', '',
           '![memory](fig_memory.png)', '', '![time](fig_time.png)', '']
    if micro:
        out += ['## Micro-benchmarks', '', '| primitive | value |', '|---|---|',
                f"| NVLink fetch-add round trip | {micro['atomic_rtt_ns_peer'] / 1e3:.2f} µs |",
                f"| local fetch-add round trip | {micro['atomic_rtt_ns_local'] / 1e3:.2f} µs |"]
        for k, v in micro.items():
            if k.startswith(('push/', 'pull/')):
                out.append(f'| {k} | {v:.0f} GB/s per GPU |')
            elif k.startswith('reserve/'):
                out.append(f"| {k} | {v['mreq_per_s']:.0f} M lane-requests/s |")
        out.append('')
    out += ['## Mooncake legacy EP', '',
            'Memory = Mooncake\'s symmetric heap (both ping-pong buffers). Per-call outputs (`packed_recv_x` etc.) '
            'are identical in both modes and not included.', '',
            rel_table(mc, 'heap_bytes', 'pair_us', 'Tiered EPArena (first design)')]
    if mcp:
        out += ['### EPA-pull (second design)', '',
                '`heap` for EPA-pull excludes the expert output `y`, which it keeps in the symmetric heap '
                '(upstream keeps the same tensor as an ordinary allocation).', '',
                rel_table([r for r in mcp if r['mode'] != 'pull (48 CTAs)'], 'heap_bytes', 'pair_us', 'EPA-pull, all SMs'),
                rel_table([r for r in mcp if r['mode'] in ('upstream', 'pull (48 CTAs)') and not r['fp8']],
                          'heap_bytes', 'pair_us', 'EPA-pull limited to 48 CTAs (upstream\'s SM count), BF16')]
    out += ['## DeepEP V2.5 (direct NVLink)', '',
            'Memory = `EPBuffer.num_bytes` (symmetric buffer, excluding the fixed 2 MB-aligned workspace).', '',
            rel_table(de, 'buffer_bytes', 'step_us', 'All runs')]
    (D / 'results.md').write_text('\n'.join(out))
    print('wrote results.md, fig_memory.png, fig_time.png')


if __name__ == '__main__':
    main()
