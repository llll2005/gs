"""Numerical test for the weighted-ADMM diagonal preconditioner (CPU, no data/GPU).

Run: python -m pytest tests/test_weighted_admm_precond.py  (or run directly)

Guards the fix for single-scalar-rho ill-conditioning in dt_admm_coordinator /
dt_admm_density_controller:
  - cond-cap prevents frozen-opacity (zero-variance) weight blow-up
  - geomean-normalization keeps the preconditioner strength-neutral (rho stays the knob)
  - per-property penalty gradients are balanced
  - no NaN/Inf through z-update + weighted dual
  - feat_std=ones reduces to the legacy unweighted path
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dt_admm_coordinator import compute_feat_std, run_gat_z_update, dual_update

N = 240
def _feats():
    torch.manual_seed(0)
    return torch.cat([torch.randn(N,3)*100, torch.randn(N,3)*1.0, torch.rand(N,2)*0.05,
                      torch.full((N,1),0.99), torch.randn(N,3)*0.3], dim=1)  # opacity frozen

def test_cond_cap_and_geomean():
    std = compute_feat_std(_feats()); w = 1.0/std**2
    assert torch.isfinite(w).all()
    assert abs(float(std.log().mean().exp()) - 1.0) < 1e-4      # geomean(std)=1
    assert float(w.max()/w.min()) < 200                          # cond capped (~100)

def test_balanced_and_strength_preserved():
    h = _feats(); std = compute_feat_std(h)
    def grads(weighted):
        hx = h.clone().requires_grad_(True)
        torch.manual_seed(1); z = h + torch.randn_like(h)*std*0.3
        ww = 1.0/std.clamp(min=1e-6)**2 if weighted else torch.ones(12)
        ((0.25)*(ww*(hx-z)**2).sum(-1).mean()).backward()
        G={"xyz":(0,3),"nrm":(3,6),"scl":(6,8),"opa":(8,9),"sh":(9,12)}
        return [float(hx.grad[:,a:b].pow(2).sum(-1).sqrt().mean()) for _,(a,b) in G.items()]
    gu, gw = grads(False), grads(True)
    import statistics as st
    assert 0.2 < st.median(gw)/st.median(gu) < 5                 # strength preserved
    assert max(gw)/min(v for v in gw if v>0) < 50                # balanced

def test_zupdate_dual_finite_and_legacy():
    h = _feats()
    graph={"node_features":h,"edge_index":torch.stack([torch.arange(N),(torch.arange(N)+1)%N]),
           "node_block_id":torch.cat([torch.zeros(N//2,dtype=torch.long),torch.ones(N//2,dtype=torch.long)]),
           "block_offsets":{0:0,1:N//2}}
    zpb,fstd = run_gat_z_update(graph,{0:torch.zeros(N//2,12),1:torch.zeros(N//2,12)},
                                rho=0.5,n_steps=10,lr=1e-3,device="cpu")
    assert torch.isfinite(torch.cat([zpb[0],zpb[1]])).all() and torch.isfinite(fstd).all()
    yn = dual_update(h[:N//2],zpb[0],torch.zeros(N//2,12),rho=0.5,feat_std=fstd)
    assert torch.isfinite(yn).all()
    # legacy: feat_std=ones == None
    assert torch.allclose(dual_update(h[:N//2],zpb[0],torch.zeros(N//2,12),rho=0.5,feat_std=None),
                          dual_update(h[:N//2],zpb[0],torch.zeros(N//2,12),rho=0.5,feat_std=torch.ones(12)))

if __name__ == "__main__":
    test_cond_cap_and_geomean(); test_balanced_and_strength_preserved(); test_zupdate_dual_finite_and_legacy()
    print("ALL TESTS PASSED ✓")
