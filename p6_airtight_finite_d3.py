# Robust and broad: find ALL finite d>=3 reps in range (k up to 12, both sectors)
# and test the targeted witness (G_eff / rho-opt). Do they fire like (4,1)?
import json
from pathlib import Path
import numpy as np
import p6_engine as E
import p6_dkj_fusion_dims as D

OUT = Path(__file__).parent

def mtilde(B, arr):
    Bd = B.conj().T; B2 = B@B; B2d = B2.conj().T
    M = 2*np.einsum('ij,njk,kl->nil', B, arr, Bd) - np.einsum('ij,njk,kl->nil', B2, arr, B2d)
    G = np.einsum('iab,jba->ij', arr, M).real
    return (G + G.T)/2

def geff_lammax(B, Q):
    d = B.shape[0]; Bd = B.conj().T; B2 = B@B; B2d = B2.conj().T
    P = (np.eye(d)+Q)/2; Pm = (np.eye(d)-Q)/2
    def DQ(X): return P@X@P - Pm@X@Pm
    M1 = DQ(Bd@Q@B)
    G = M1 + Bd@M1@B - DQ(B2d@Q@B2)
    return float(np.linalg.eigvalsh((G + G.conj().T)/2)[-1])

def best_over_Q(B, arr, rng, n_rand=200):
    d = B.shape[0]; best = -9.0
    G = mtilde(B, arr); w, V = np.linalg.eigh(G)
    for idx in range(len(w)-1, max(len(w)-4, -1), -1):
        Qs = np.tensordot(V[:, idx], arr, axes=(0, 0)); ev, Uu = np.linalg.eigh(Qs); order = np.argsort(ev)
        for r in range(1, d):
            s = np.ones(d); s[order[:r]] = -1.0
            best = max(best, geff_lammax(B, (Uu*s) @ Uu.conj().T))
    for _ in range(n_rand):
        for r in range(1, d):
            Mm = rng.normal(size=(d, r)) + 1j*rng.normal(size=(d, r)); Vv, _ = np.linalg.qr(Mm)
            best = max(best, geff_lammax(B, 2*(Vv@Vv.conj().T) - np.eye(d)))
    return best

def sector(k, twoj, which):
    try:
        res = D.largest(k, twoj) if which == "L" else D.smallest_nontrivial(k, twoj)
        if res is None: return None
        twoJ, d = res
        gen = E.braid_generators(k, twoj, twoJ)
        return gen["d"], gen
    except Exception:
        return None

CAP = 2500
print("=== all finite d>=3 reps (k<=12, both sectors) + targeted witness ===\n")
seen = set(); finite_d3 = []
for k in range(2, 13):
    for twoj in range(2, min(2*k, 7)+1):            # j=1..3 (2j=2..6), j<=k/2
        for which in ("L", "S"):
            s = sector(k, twoj, which)
            if s is None: continue
            d, gen = s
            if d < 3: continue
            key = (k, twoj, d)
            if key in seen: continue
            seen.add(key)
            Graw = E.enumerate_group(gen["s1"], gen["s2"], max_size=CAP)
            if Graw is None or len(Graw) >= CAP: continue   # dense/open
            finite_d3.append((k, twoj, d, gen, Graw))

print(f"finite d>=3 reps found: {len(finite_d3)}")
jl = {2: "1", 3: "3/2", 4: "2", 5: "5/2", 6: "3"}
fired = []
rows = []
for k, twoj, d, gen, Graw in finite_d3:
    arr = list(np.stack(E.herm_basis(d)))
    rng = np.random.default_rng(7*k + twoj)
    best = max(best_over_Q(E.to_sud(B), arr, rng) for B in Graw)
    f = best >= 1.49
    if f: fired.append((k, jl.get(twoj, twoj)))
    rows.append({"k": k, "j": jl.get(twoj, twoj), "d": d, "group_order": len(Graw),
                 "targeted_max": best, "fires": f})
    print(f"  ({k},{jl.get(twoj,twoj)}) d={d} FINITE |G|={len(Graw):4d}: targeted-max={best:.5f}  {'-> FIRES' if f else '-> does NOT fire'}")

print(f"\n  finite d>=3 tested: {len(finite_d3)} | fire (targeted): {len(fired)} -> {fired}")
if len(fired) >= 2:
    print("  => pattern confirmed (n>1): the targeted witness fires for finite d>=3 reps, not only (4,1).")
elif len(finite_d3) <= 1:
    print("  => only 1 finite d>=3 rep in range -> n>1 from THIS family not possible; an alternative argument is needed.")
else:
    print("  => unexpected -> clarify before drawing a conclusion.")

out = {
    "meta": {"script": "p6_airtight_finite_d3.py", "k_range": [2, 12], "fire_threshold": 1.49},
    "rows": rows,
    "fired": [{"k": k, "j": j} for k, j in fired],
    "n_finite_d3_tested": len(finite_d3),
    "n_fired": len(fired),
}
(OUT / "p6_airtight_finite_d3_results.json").write_text(json.dumps(out, indent=2))
print(f"\nWRITE {OUT / 'p6_airtight_finite_d3_results.json'}")
