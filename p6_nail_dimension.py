# Is the EXTRA DIMENSION the lever (not the angle)?
# Same 2-block rotation Ry(theta), once at d=2 (B=Ry), once at d=3 (B=Ry (+) trivial). Targeted witness.
# Expectation if "d>=3 spectral freedom" holds:
#   60deg: d=2 -> 1.5 (60 is the d=2 optimum)       d=3 -> 1.5
#   72deg: d=2 -> 1.427 (72 misses the optimum)     d=3 -> 1.5  <== the extra dim lifts it, despite 72deg
# => at d=2 angle-dependent, at d=3 angle-INDEPENDENT -> the extra dimension is the lever.
import json
from pathlib import Path
import numpy as np
import p6_engine as E

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

def best_over_Q(B, arr, rng, n_rand=400):
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

def Ry(th): c, s = np.cos(th/2), np.sin(th/2); return np.array([[c, -s], [s, c]], dtype=complex)
def blk1(M): d = M.shape[0]+1; R = np.eye(d, dtype=complex); R[:M.shape[0], :M.shape[0]] = M; return R

arr2 = list(np.stack(E.herm_basis(2))); arr3 = list(np.stack(E.herm_basis(3)))
print("=== extra dimension vs angle (targeted witness, single B) ===\n")
rows = []
for deg, th in [(60, np.pi/3), (72, 2*np.pi/5), (90, np.pi/2), (100, 100*np.pi/180)]:
    B2 = Ry(th); B3 = blk1(Ry(th))
    v2 = best_over_Q(B2, arr2, np.random.default_rng(deg))
    v3 = best_over_Q(B3, arr3, np.random.default_rng(deg+1))
    rows.append({"degrees": deg, "d2_targeted_max": v2, "d2_fires": bool(v2 >= 1.49),
                 "d3_targeted_max": v3, "d3_fires": bool(v3 >= 1.49)})
    print(f"  Ry({deg:3d}deg): d=2 -> {v2:.5f} {'FIRES' if v2>=1.49 else 'no   '}   |   "
          f"d=3(+triv) -> {v3:.5f} {'FIRES' if v3>=1.49 else 'no'}")
print("\n  reading confirmed if: d=2 angle-DEPENDENT (60 fires, 72/90/100 do not), d=3 ALWAYS fires.")
print("  => the EXTRA DIMENSION is the lever, not the angle.")

out = {"meta": {"script": "p6_nail_dimension.py", "fire_threshold": 1.49}, "rows": rows}
(OUT / "p6_nail_dimension_results.json").write_text(json.dumps(out, indent=2))
print(f"\nWRITE {OUT / 'p6_nail_dimension_results.json'}")
