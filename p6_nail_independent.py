# Independent check -- own rho-opt solver.
# NO import of p6_engine, NO best_over_Q, NOT the einsum G_eff form.
# A completely different code path: direct sequential Luders measurement statistics.
#   K3(rho) = C12 + C23 - C13 (3-time LGI), each correlator derived from measurement probabilities:
#     C12 = sum_{s1,s2} s1 s2 Tr[Pi_s2 B Pi_s1 rho Pi_s1 B^dag]            (measure at t1,t2)
#     C23 = sum_{s2,s3} s2 s3 Tr[Pi_s3 B Pi_s2 (B rho B^dag) Pi_s2 B^dag]  (measure at t2,t3; rho2=B rho B^dag)
#     C13 = sum_{s1,s3} s1 s3 Tr[Pi_s3 B^2 Pi_s1 rho Pi_s1 B^2^dag]        (measure at t1,t3, NONE at t2)
#   -> rewritten as K3(rho)=Tr[rho M], M = E12 + E23 - E13 (witness operator).
#   rho-opt = lambda_max(sym(M)) (max over density matrices). Q-opt = independent dense search.
# Reduces at rho=I/d to Tr[M]/d (= neutral ceiling). Expectation:
#   d=2: rho-opt ANGLE-dependent (60->1.5, 72->1.427, 90->1.0, 100->1.0)
#   d=3(+triv): rho-opt ALWAYS 1.5; neutral ceiling (3d-1)/(2d) = 4/3 = 1.333 (artifact, bound-consistent)
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def projectors(Q):
    I = np.eye(Q.shape[0], dtype=complex)
    return (I + Q) / 2, (I - Q) / 2

def witness_operator(B, Q):
    """K3(rho) = Tr[rho M], with M derived from the sequential Luders statistics (see above)."""
    d = B.shape[0]; Bd = B.conj().T; B2 = B @ B; B2d = B2.conj().T
    Pp, Pm = projectors(Q); Pi = {1: Pp, -1: Pm}
    E1 = np.zeros((d, d), dtype=complex)   # one-step correlator operator (C12)
    E13 = np.zeros((d, d), dtype=complex)  # two-step (t2 skipped) (C13)
    for a in (1, -1):
        for b in (1, -1):
            E1  += a * b * (Pi[a] @ Bd  @ Pi[b] @ B  @ Pi[a])
            E13 += a * b * (Pi[a] @ B2d @ Pi[b] @ B2 @ Pi[a])
    E12 = E1
    E23 = Bd @ E1 @ B                      # Tr[rho2 E1] = Tr[rho B^dag E1 B], rho2 = B rho B^dag
    return E12 + E23 - E13

def random_dichotomic(d, r, rng):
    """random dichotomic Q: U diag(r times -1, rest +1) U^dag, U Haar-random."""
    M = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    U, _ = np.linalg.qr(M)
    diag = np.ones(d); diag[:r] = -1.0
    return (U * diag) @ U.conj().T

def bloch_Q(th, ph):
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return n[0] * X + n[1] * Y + n[2] * Z

def scan(B, rng, n_samples, grid_d2=False):
    d = B.shape[0]; best_rho = -9.0; best_neu = -9.0
    cand = []
    if grid_d2 and d == 2:                 # d=2: quasi-exhaustive Bloch grid search
        for th in np.linspace(0, np.pi, 120):
            for ph in np.linspace(0, 2 * np.pi, 120):
                cand.append(bloch_Q(th, ph))
    for r in range(1, d):                  # rank of the -1 eigenspace: 1..d-1
        for _ in range(n_samples):
            cand.append(random_dichotomic(d, r, rng))
    for Q in cand:
        M = witness_operator(B, Q)
        Msym = (M + M.conj().T) / 2
        best_rho = max(best_rho, float(np.linalg.eigvalsh(Msym)[-1]))
        best_neu = max(best_neu, float(np.trace(M).real) / d)
    return best_neu, best_rho

def Ry(th):
    c, s = np.cos(th / 2), np.sin(th / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def embed_triv(M2, d):
    R = np.eye(d, dtype=complex); R[:2, :2] = M2; return R

print("=== independent check (own solver, sequential Luders statistics) ===")
print("    [neutral=rho=I/d ceiling | rho-opt=max over rho]   FIRES := rho-opt >= 1.49\n")
print(f"  {'angle':>8} | {'d=2 neutral':>11}  {'d=2 rho-opt':>11} | {'d=3 neutral':>11}  {'d=3 rho-opt':>11}")
print("  " + "-" * 70)
rows = []
for deg, th in [(60, np.pi / 3), (72, 2 * np.pi / 5), (90, np.pi / 2), (100, 100 * np.pi / 180)]:
    B2 = Ry(th); B3 = embed_triv(Ry(th), 3)
    n2, r2 = scan(B2, np.random.default_rng(1000 + deg), 1500, grid_d2=True)
    n3, r3 = scan(B3, np.random.default_rng(2000 + deg), 4000)
    f2 = "FIRES" if r2 >= 1.49 else "no   "
    f3 = "FIRES" if r3 >= 1.49 else "no   "
    rows.append({"degrees": deg, "d2_neutral": n2, "d2_rho_opt": r2, "d2_fires": bool(r2 >= 1.49),
                 "d3_neutral": n3, "d3_rho_opt": r3, "d3_fires": bool(r3 >= 1.49)})
    print(f"  {deg:>6}deg | {n2:>11.5f}  {r2:>9.5f} {f2} | {n3:>11.5f}  {r3:>9.5f} {f3}")
print("\n  Confirmed if: d=2 rho-opt ANGLE-dependent (only 60 fires), d=3 rho-opt ALWAYS fires (all 1.5),")
print("                d=3 neutral ceiling ~1.333 = (3d-1)/(2d) (artifact, bound-consistent).")
print("  -> the EXTRA DIMENSION is the lever. Does the independent code path reproduce this? yes/no.")

out = {"meta": {"script": "p6_nail_independent.py", "fire_threshold": 1.49,
                "note": "independent code path: sequential Luders measurement statistics, no p6_engine import"},
       "rows": rows}
(OUT / "p6_nail_independent_results.json").write_text(json.dumps(out, indent=2))
print(f"\nWRITE {OUT / 'p6_nail_independent_results.json'}")
