"""Verify the cloudops eval pad-fix: with a model built at N=SPLIT+PL, feeding a
context padded to N with an arbitrary tail must leave hid[:, SPLIT-1] unchanged
(causal mask), and the padded call must match the training-time input shape."""
import sys, os, torch, numpy as np
sys.path.insert(0, "/share/hpcproject/zzhang66/nfm/code/common")
from nfm_v2 import HSTUv2, GRUv2, ContinuousInput, HorizonHead

SPLIT, PL = 288, 48; N = SPLIT + PL
D, H, L, KC, MIX = 64, 4, 2, 16, 2
torch.manual_seed(0)

def build(kind):
    inp = ContinuousInput(D, KC); head = HorizonHead(D, MIX)
    return (HSTUv2 if kind == "HSTU" else GRUv2)(inp, head, D=D, H=H, L=L, N=N).eval()

for kind in ["HSTU", "GRU"]:
    m = build(kind)
    z_ctx = torch.randn(2, SPLIT); c_ctx = torch.randint(0, KC, (2, SPLIT))
    t = torch.zeros(2, N, dtype=torch.long)
    za = torch.cat([z_ctx, torch.zeros(2, PL)], 1)
    zb = torch.cat([z_ctx, torch.randn(2, PL)], 1)
    ca = torch.cat([c_ctx, torch.zeros(2, PL, dtype=torch.long)], 1)
    cb = torch.cat([c_ctx, torch.randint(0, KC, (2, PL))], 1)
    with torch.no_grad():
        ha = m((za, ca), t)[:, SPLIT - 1]
        hb = m((zb, cb), t)[:, SPLIT - 1]
    assert torch.allclose(ha, hb, atol=1e-5), f"{kind}: tail leaks into hid[SPLIT-1]"
    print(f"{kind}: pad-causality OK  (max diff {(ha-hb).abs().max().item():.2e})")

# repro of the original crash: unpadded 288-length input must fail on HSTU
m = build("HSTU")
try:
    with torch.no_grad():
        m((torch.randn(2, SPLIT), torch.randint(0, KC, (2, SPLIT))), torch.zeros(2, SPLIT, dtype=torch.long))
    print("WARNING: unpadded input did NOT crash (mask now dynamic?)")
except RuntimeError as e:
    print(f"unpadded 288 input still crashes as expected: {str(e)[:80]}")
print("PAD_TEST_OK")
