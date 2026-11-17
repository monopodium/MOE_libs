"""epa.pull_rows: rows land in order at the device offset; run flag 0 does nothing. Run: python epa/tests/test_pull.py (1 GPU)"""
import sys, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'python'))
from epa import pull_rows
D = 7168
src = torch.randn(5000, D, device='cuda').bfloat16()
perm = torch.randperm(5000, device='cuda')[:3000]
ptrs = src.data_ptr() + perm.to(torch.int64) * D * 2
buf = torch.zeros(3000 * D * 2 + 4096, dtype=torch.uint8, device='cuda')
off = torch.tensor([2048], device='cuda', dtype=torch.int64)
for run, rows in ((0, 3000), (1, 2500)):
    buf.zero_()
    pull_rows(buf, off, ptrs, torch.tensor([rows], device='cuda', dtype=torch.int32), D * 2,
              run_1=torch.tensor([run], device='cuda', dtype=torch.int64))
    got = buf[2048:2048 + 3000 * D * 2].view(torch.bfloat16).view(3000, D)
    if run:
        assert torch.equal(got[:rows], src[perm[:rows]]) and not got[rows:].any()
    else:
        assert not buf.any()
print('pull ok')
