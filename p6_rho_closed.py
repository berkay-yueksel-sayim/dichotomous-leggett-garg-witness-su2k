"""
Closed-form: the rho=I/d ceiling is a resolution artifact; a targeted (true-LGI) witness -> 3/2 for any d.
Exact matrices.

Background (Budroni-Emary, arXiv:1309.3678): max K3 under Luders measurement with a dichotomic Q is
3/2, dimension-INDEPENDENT, via rho optimization. A witness that fixes rho=I/d gives, for odd d, the
diluted value (3d-1)/(2d).

Important subtlety: K3=2C(B)-C(B^2) holds ONLY for STATIONARY rho (C12=C23 requires B rho B^dag=rho).
For general rho one must use the true sequential-LGI K3 (C23 uses the EVOLVED state B rho B^dag).
Naively maximizing 2C-C over all rho gives an unphysical 2.98 > 3/2. Correct form:
  C12=Tr[B^dag Q B D_Q(rho)], C23=Tr[B^dag Q B D_Q(B rho B^dag)], C13=Tr[B^2^dag Q B^2 D_Q(rho)],
  D_Q(X)=Pi+ X Pi+ - Pi- X Pi-.
  K3(rho)=Tr[rho G_eff],  G_eff = D_Q(B^dag Q B) + B^dag D_Q(B^dag Q B) B - D_Q(B^2^dag Q B^2).
  rho=I/d -> (1/d)Tr[G_eff]   (the neutral witness; rho=I/d is stationary, so 2C-C holds).
  rho-opt -> lambda_max(sym(G_eff)) <= 3/2.   (true LGI, ALL rho; not lambda_max(sym(V))!)

Construction (m=floor(d/2) saturating 2-dim blocks + leftover level), uses C=R_zz, B2=rotation alpha=pi/3:
  rho=I/d -> (3d-1)/(2d) [ODD] / 3/2 [EVEN];  rho-opt -> 3/2 for any d  -> the ceiling is a rho=I/d artifact.
"""
import numpy as np
import json
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
alpha = np.pi/3.0
B2 = np.cos(alpha/2)*I2 - 1j*np.sin(alpha/2)*sx
Q2 = sz


def DQ(X, Q):
    d = len(Q); Pp = (np.eye(d) + Q)/2; Pm = (np.eye(d) - Q)/2
    return Pp @ X @ Pp - Pm @ X @ Pm


def G_eff(B, Q):
    """K3(rho)=Tr[rho G_eff] (true sequential LGI, arbitrary rho)."""
    Bd = B.conj().T; B2 = B @ B; B2d = B2.conj().T
    A = Bd @ Q @ B; A2 = B2d @ Q @ B2
    return DQ(A, Q) + Bd @ DQ(A, Q) @ B - DQ(A2, Q)


def k3_rho(B, Q, rho):
    G = G_eff(B, Q)
    return float(np.real(np.trace(rho @ (G + G.conj().T)/2)))


def k3_rhoId(B, Q):
    d = len(Q); G = G_eff(B, Q)
    return float(np.real(np.trace(G))/d)          # rho=I/d (stationary)


def k3_rhoopt(B, Q):
    G = G_eff(B, Q)
    return float(np.linalg.eigvalsh((G + G.conj().T)/2)[-1])   # true LGI rho-opt, <=3/2


# ---------- d=2 block ----------
Bc2 = B2.copy(); Qc2 = Q2.copy()
print("=== closed-form: rho=I/d ceiling is an artifact; targeted LGI witness -> 3/2 (exact, no sampling) ===\n")
print(f"d=2 block (B2=rotation pi/3, Q=Z): rho=I/d={k3_rhoId(Bc2,Qc2):.7f}  LGI-rho-opt={k3_rhoopt(Bc2,Qc2):.7f}  (target 3/2)\n")


def embed(d):
    m = d // 2
    B = np.eye(d, dtype=complex); Q = np.eye(d, dtype=complex)
    for b in range(m):
        B[2*b:2*b+2, 2*b:2*b+2] = B2
        Q[2*b:2*b+2, 2*b:2*b+2] = Q2
    if d % 2 == 1:
        Q[d-1, d-1] = 1.0
    return B, Q


print(f"{'d':>3} {'parity':>8} {'rho=I/d':>14} {'ceiling(par.)':>13} {'LGI-rho-opt':>11} {'<=3/2 & opt=3/2':>15}")
rows = []
for d in (2, 3, 4, 5, 6, 7):
    B, Q = embed(d)
    kmix = k3_rhoId(B, Q)
    kopt = k3_rhoopt(B, Q)
    cap = 1.5 if d % 2 == 0 else (3*d-1)/(2*d)
    par = "even" if d % 2 == 0 else "odd"
    ok = abs(kmix - cap) < 1e-9 and abs(kopt - 1.5) < 1e-9
    print(f"{d:>3} {par:>8} {kmix:>14.7f} {cap:>13.7f} {kopt:>11.7f} {('OK' if ok else 'X'):>15}")
    rows.append(dict(d=d, parity=par, K3_rho_Id=float(kmix), cap_parity=float(cap),
                     K3_LGI_rhoopt=float(kopt), match=bool(ok)))

allok = all(r["match"] for r in rows)
print(f"\n  rho=I/d (neutral witness): (3d-1)/(2d) [odd] / 3/2 [even]  (matches robust lower bounds: 4/3, 3/2, 7/5).")
print(f"  LGI-rho-opt (targeted witness, true sequential LGI): = 3/2 for every d (<=3/2 bound, attained here).")
print(f"  -> the ceiling is a rho=I/d artifact; the targeted witness lifts it to 3/2. ALL match: {allok}")
print(f"  note: lambda_max(sym(V)) (over all rho, NON-stationary) gave an unphysical ~3.0 - wrong; the true")
print(f"  LGI-rho-opt (G_eff, evolved state in C23) is <=3/2. rho=I/d and the rho-block are stationary (valid).")

with open(os.path.join(HERE, "p6_rho_closed.json"), "w") as fh:
    json.dump(dict(note="closed (exact): rho=I/d=(3d-1)/(2d) odd /3/2 even; true sequential-LGI rho-opt (G_eff)=3/2 any d",
                   correction="2C-C valid only for stationary rho; general rho needs evolved-state C23 (G_eff); "
                              "lambda_max(sym(V)) over all rho gives unphysical ~3.0 (non-stationary) - WRONG",
                   budroni_emary="max K3 dichotomic Lueders=3/2 dim-independent via rho-opt (1309.3678)",
                   all_match=bool(allok), rows=rows), fh, indent=2)
print("\n>>> closed-form. neutral rho=I/d -> parity-dependent ceiling; targeted (true LGI) -> 3/2 for any d. <<<")
