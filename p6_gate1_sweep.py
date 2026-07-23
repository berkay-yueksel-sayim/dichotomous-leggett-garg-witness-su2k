"""
(k, j) density-witness sweep.

Engine validated separately (p6_gate0_validate.py: gate-0 reproduces the d=2 reference,
Yang-Baxter ~1e-15 for d=2..5). Per (k, j, sector):
(1) density status via enumeration (closes = finite / does not = dense);
(2) tier-1 = max_B lambda_max(M-tilde(B)) (outer enumeration/sampling, inner closed-form);
(3) tier-2 ONLY where tier-1 >= 3/2 - tol = explicit dichotomic-Q optimization -> witness value F.
Output columns: d; dense?; lambda_max(M-tilde) [= max-over-Q quantity];
saturates? (F >= 3/2 - tol); agreement.

Two additions, both leaving the reported quantity UNCHANGED:
  (i)  enumerate_group_fast: hash de-duplication O(n^2)->O(n). Cross-verified against the
       O(n^2) reference (E.enumerate_group) on k=4 and k=8 -> identical group + identical
       lambda_max + identical Q=Z K3.
  (ii) Q=Z diagnostic column (d=2 only): max K3 with fixed Q=Z=diag(1,-1). Certifies the
       j=1/2 row (k=4 -> 1, k=8 -> 3/sqrt5, dense -> ~1.4999). The reported quantity stays
       max-over-Q; Q=Z is purely a consistency / sampling-adequacy diagnostic.

Grid: j=1/2 k2..10 (smallest sector), j=1 k4..10 (smallest), j=3/2 k6..10 (largest), j=2 k8..10 (largest).
"""
import numpy as np
import json
import os
import sys
import time
from collections import defaultdict
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import p6_engine as E
import p6_dkj_fusion_dims as D

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
SEED = 20260602
LUEDERS = 1.5
TOL = 1e-2              # saturates: F >= 3/2 - TOL (documented; raw values also reported)
DICHTE_MAXSIZE = 3000
L_EX = 6               # exhaustive braid word length (dense cases)
N_RAND = 20000         # total random braid words (dense cases), spread over several lengths
RAND_LENS = (10, 16, 22, 30)   # several lengths (better saturation, same budget)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)   # reference observable, d=2 only
SQRT3_5 = 3.0/np.sqrt(5.0)     # = 1.3416407865 (Q=Z, k=8); gate-0 anchor
NREST = {2: 1500, 3: 3000, 4: 4500, 5: 6000}   # tier-2 random restarts per d (eigen-seed carries most of the load)
CONV_TOL = 5e-3                 # tier-2 plateau criterion: F(3n) - F(n) < CONV_TOL
FOURTHIRDS = 4.0/3.0           # tier-2 gate-0 anchor: exact dichotomic F of (4,1) d=3


# ---------- hash enumeration O(n) (cross-verified against E.enumerate_group O(n^2)) ----------
def enumerate_group_fast(s1, s2, tol=1e-8, max_size=DICHTE_MAXSIZE, scale=1e4):
    """Identical to E.enumerate_group, only the de-duplication uses hash buckets plus an exact
    bucket check. The bucket check (np.max|U-V|<tol) prevents a false merge; the coarse grid
    rounding (1e-4) prevents a false split. Correctness against O(n^2) shown by ab_verify() on k=4+k=8."""
    Id = np.eye(s1.shape[0], dtype=complex)
    gens = [E.to_sud(s1), E.to_sud(s1.conj().T), E.to_sud(s2), E.to_sud(s2.conj().T)]

    def fp(U):
        v = np.concatenate([U.real.ravel(), U.imag.ravel()])
        return tuple(np.round(v*scale).astype(np.int64))

    buckets = defaultdict(list)
    elems = []

    def add(U):
        buckets[fp(U)].append(U)
        elems.append(U)

    def seen(U):
        for V in buckets[fp(U)]:
            if np.max(np.abs(U - V)) < tol:
                return True
        return False

    add(Id)
    frontier = [Id]
    for g in gens:
        if not seen(g):
            add(g); frontier.append(g)
    while frontier:
        nf = []
        for U in frontier:
            for g in gens:
                W = E.to_sud(U @ g)
                if not seen(W):
                    add(W); nf.append(W)
                    if len(elems) > max_size:
                        return None
        frontier = nf
    return elems


# ---------- M-tilde witness bound (max-over-Q, closed form) ----------
def gram_Mtilde(B, arr):
    """Gram matrix of sym(2 Phi_B - Phi_{B^2}) in the orthonormal Hermitian basis arr (n,d,d), symmetrized."""
    Bd = B.conj().T; B2 = B @ B; B2d = B2.conj().T
    BEBd = np.einsum('ij,njk,kl->nil', B, arr, Bd)
    B2EB2d = np.einsum('ij,njk,kl->nil', B2, arr, B2d)
    M = 2*BEBd - B2EB2d
    G = np.einsum('iab,jba->ij', arr, M).real
    return (G + G.T)/2


def mtilde_lammax(B, arr):
    """lambda_max(M-tilde(B)) = max-over-Q K3 (reported quantity, closed-form eigenvalue)."""
    return float(np.linalg.eigvalsh(gram_Mtilde(B, arr))[-1])


def _C_batched_Z(U):
    """C(U)=0.5 Re Tr[Z U Z U^dag] batched, d=2, Q=Z=diag(1,-1) (reference protocol). U:(n,2,2)."""
    M = Z2 @ U @ Z2 @ np.conj(np.transpose(U, (0, 2, 1)))
    return 0.5*np.real(M[:, 0, 0] + M[:, 1, 1])


def qz_deep_d2(s1, s2, finite_group, seed, Lex=9, rand_lens=(12, 16, 20, 24, 28, 32, 36, 40), nsamp=200000):
    """Q=Z diagnostic (d=2 only) at full sampling depth, vectorized (saturates near 1.49996).
    max_B K3(B,Q=Z), K3=2C(B)-C(B^2). finite_group=exact / None=deep-sampled. Not the reported quantity."""
    if finite_group is not None:
        W = np.stack(finite_group)
        K3 = 2*_C_batched_Z(W) - _C_batched_Z(np.matmul(W, W))
        return float(K3.max())
    gens = np.stack([E.to_sud(s1), E.to_sud(s1.conj().T), E.to_sud(s2), E.to_sud(s2.conj().T)])
    best = -np.inf
    words = gens.copy()
    for L in range(1, Lex+1):
        K3 = 2*_C_batched_Z(words) - _C_batched_Z(np.matmul(words, words))
        best = max(best, float(K3.max()))
        if L < Lex:
            words = np.matmul(words[:, None, :, :], gens[None, :, :, :]).reshape(-1, 2, 2)
    rng = np.random.default_rng(seed)
    for Lr in rand_lens:
        idx = rng.integers(0, 4, size=(nsamp, Lr))
        W = np.tile(np.eye(2, dtype=complex), (nsamp, 1, 1))
        for j in range(Lr):
            W = np.matmul(W, gens[idx[:, j]])
        K3 = 2*_C_batched_Z(W) - _C_batched_Z(np.matmul(W, W))
        best = max(best, float(K3.max()))
    return best


def outer_max_tier1(s1, s2, arr, finite_group, qz_Z=None):
    """max_B lambda_max(M-tilde) (reported). finite_group=list (exact) or None (dense -> sample).
    qz_Z (d=2): additionally max_B K3(B,Q=Z) over the SAME set of words -> Q=Z diagnostic column.
    Returns: (lam_max, bestB, top15_B, qz_max | None)."""
    qz_max = -np.inf if qz_Z is not None else None
    if finite_group is not None:
        vals = []
        for B in finite_group:
            vals.append(mtilde_lammax(B, arr))
            if qz_Z is not None:
                f = E.K3_Qfixed(B, qz_Z)
                if f > qz_max:
                    qz_max = f
        vals = np.array(vals)
        order = np.argsort(vals)
        bestB = finite_group[int(order[-1])]
        top = [finite_group[i] for i in order[-40:]]
        return float(vals.max()), bestB, top, (float(qz_max) if qz_Z is not None else None)
    # dense: exhaustive L<=L_EX + random over several lengths
    d = s1.shape[0]
    gens = np.stack([E.to_sud(s1), E.to_sud(s1.conj().T), E.to_sud(s2), E.to_sud(s2.conj().T)])
    best = -np.inf; bestB = None; top = []

    def consider(B):
        nonlocal best, bestB, qz_max
        v = mtilde_lammax(B, arr)
        if v > best:
            best, bestB = v, B
        top.append((v, B))
        if qz_Z is not None:
            f = E.K3_Qfixed(B, qz_Z)
            if f > qz_max:
                qz_max = f

    words = gens.copy()
    for L in range(1, L_EX+1):
        for B in words:
            consider(B)
        if L < L_EX:
            words = np.matmul(words[:, None, :, :], gens[None, :, :, :]).reshape(-1, d, d)
    rng = np.random.default_rng(SEED + d)
    per = max(1, N_RAND // len(RAND_LENS))
    for Lr in RAND_LENS:
        for _ in range(per):
            idx = rng.integers(0, 4, size=Lr)
            B = np.eye(d, dtype=complex)
            for t in idx:
                B = B @ gens[t]
            consider(B)
    top.sort(key=lambda x: x[0])
    return float(best), bestB, [b for _, b in top[-40:]], (float(qz_max) if qz_Z is not None else None)


def _best_dich_for_B(B, arr, d, n_restart, rng):
    """max_Q over dichotomic K3(B,Q). Eigen-seed (M-tilde top eigenvectors -> +/-1, all ranks)
    plus random restarts. The eigen-seed carries most of the load (it rounds the generally
    non-dichotomic optimal Q* to the nearest dichotomic Q; random restarts refine)."""
    Bd = B.conj().T; B2 = B @ B; B2d = B2.conj().T

    def K3(Q):
        return (2*np.trace(Q@B@Q@Bd).real - np.trace(Q@B2@Q@B2d).real)/d

    best = -np.inf
    # 1) eigen-seed: optimal Q* from M-tilde Gram top eigenvectors, rounded to +/-1 (all rank cuts)
    w, V = np.linalg.eigh(gram_Mtilde(B, arr))
    for idx in range(len(w)-1, max(len(w)-4, -1), -1):
        Qstar = np.tensordot(V[:, idx], arr, axes=(0, 0))   # = sum_i V[i,idx] arr_i (Hermitian)
        ev, U = np.linalg.eigh(Qstar)
        for r in range(1, d):
            signs = np.ones(d); signs[np.argsort(ev)[:r]] = -1.0
            best = max(best, K3((U*signs) @ U.conj().T))
        best = max(best, K3((U*np.sign(ev)) @ U.conj().T))
    # 2) random restarts over all ranks
    for r in range(1, d):
        for _ in range(n_restart // (d-1)):
            Mm = rng.normal(size=(d, r)) + 1j*rng.normal(size=(d, r))
            Vv, _ = np.linalg.qr(Mm)
            best = max(best, K3(2*(Vv @ Vv.conj().T) - np.eye(d)))
    return best


def tier2_hard(Bs, arr, d, n_restart, seed):
    """Witness value F = max over B in Bs [max over dichotomic Q]. Per-B deterministic seed:
    n and 3n share the same stream per B, which guarantees F(3n) >= F(n) for the plateau check."""
    best = -np.inf
    for bi, B in enumerate(Bs):
        rng = np.random.default_rng(seed + 1009*bi)
        best = max(best, _best_dich_for_B(B, arr, d, n_restart, rng))
    return float(best)


def tier2_gate0():
    """Gate-0 anchor (guard 2): tier-2 MUST reproduce the exact dichotomic F of
    (4,1) d=3 = 4/3 (over ALL group elements B). Otherwise there is a solver error -> stop."""
    twoJ, _ = D.smallest_nontrivial(4, 2)        # j=1, k=4 -> 2J=2, d=3
    gen = E.braid_generators(4, 2, twoJ)
    Gall = enumerate_group_fast(gen["s1"], gen["s2"], max_size=4000)
    arr = np.stack(E.herm_basis(3))
    F = tier2_hard(Gall, arr, 3, NREST[3], SEED)
    ok = abs(F - FOURTHIRDS) < 3e-3
    print(f"=== tier-2 gate-0 (guard 2): tier-2 vs exact F((4,1),d=3)=4/3 ===")
    print(f"  (4,1) d=3 |G|={len(Gall)} -> F={F:.6f}  vs  4/3={FOURTHIRDS:.6f}  "
          f"-> {'PASS' if ok else 'FAIL - solver error!'}\n")
    return ok, float(F)


# ---------- cross-verification: enumerate_group_fast == E.enumerate_group ----------
def ab_verify():
    print("=== cross-verification: enumerate_group_fast vs E.enumerate_group O(n^2) ===")
    ok = True
    for k in (4, 8):   # finite j=1/2 anchors -- require identical group + identical K3
        twoJ, _ = D.smallest_nontrivial(k, 1)
        gen = E.braid_generators(k, 1, twoJ)
        s1, s2 = gen["s1"], gen["s2"]
        t0 = time.perf_counter(); Gs = E.enumerate_group(s1, s2, max_size=DICHTE_MAXSIZE); ts = time.perf_counter()-t0
        t0 = time.perf_counter(); Gf = enumerate_group_fast(s1, s2, max_size=DICHTE_MAXSIZE); tf = time.perf_counter()-t0
        same_size = (Gs is not None and Gf is not None and len(Gs) == len(Gf))
        same_set = same_size and all(any(np.max(np.abs(A-Bm)) < 1e-7 for Bm in Gs) for A in Gf)
        arr = np.stack(E.herm_basis(2))
        lam_s = max(mtilde_lammax(B, arr) for B in Gs)
        lam_f = max(mtilde_lammax(B, arr) for B in Gf)
        qz_s = max(E.K3_Qfixed(B, Z2) for B in Gs)
        qz_f = max(E.K3_Qfixed(B, Z2) for B in Gf)
        match = bool(same_set and abs(lam_s-lam_f) < 1e-9 and abs(qz_s-qz_f) < 1e-9)
        ok = ok and match
        print(f"  k={k}: |G| slow/fast = {len(Gs)}/{len(Gf)} same_set={same_set} | "
              f"max-Q {lam_s:.9f}/{lam_f:.9f} | Q=Z {qz_s:.9f}/{qz_f:.9f} | "
              f"{'IDENTICAL' if match else 'MISMATCH!!'} | t {ts*1000:.0f}/{tf*1000:.0f} ms")
    # dense speed proof (capped at 800 -- both must return 'dense'=None)
    twoJ, _ = D.smallest_nontrivial(7, 1)
    gen = E.braid_generators(7, 1, twoJ); s1, s2 = gen["s1"], gen["s2"]
    t0 = time.perf_counter(); Gs = E.enumerate_group(s1, s2, max_size=800); ts = time.perf_counter()-t0
    t0 = time.perf_counter(); Gf = enumerate_group_fast(s1, s2, max_size=800); tf = time.perf_counter()-t0
    dense_ok = (Gs is None) and (Gf is None)
    print(f"  k=7 (dense, cap 800): slow None={Gs is None} ({ts:.2f}s) / fast None={Gf is None} ({tf*1000:.0f}ms) "
          f"| ~{ts/max(tf,1e-9):.0f}x | both dense: {dense_ok}")
    verdict = ok and dense_ok
    print(f"  -> cross-check {'PASS - fast IDENTICAL to O(n^2) on anchors, dense consistent.' if verdict else 'FAIL - do NOT use fast!'}\n")
    return verdict


# ---------- grid ----------
GRID = []
for k in range(2, 11):
    GRID.append((k, 1, "klein"))           # j=1/2
for k in range(4, 11):
    GRID.append((k, 2, "klein"))           # j=1
for k in range(6, 11):
    GRID.append((k, 3, "gross"))           # j=3/2
for k in range(8, 11):
    GRID.append((k, 4, "gross"))           # j=2

print("=== (k,j) sweep -- correspondence table ===")
print(f"    [tol={TOL}, Luders={LUEDERS}, density-bound={DICHTE_MAXSIZE}, sample L<={L_EX} + {N_RAND}@{RAND_LENS}]\n")

AB = ab_verify()
if not AB:
    print(">>> cross-verification FAIL -- hash speedup deviates from O(n^2). STOP, no sweep. <<<")
    sys.exit(1)

T2G0, t2g0_F = tier2_gate0()
if not T2G0:
    print(">>> tier-2 gate-0 FAIL -- tier-2 does not hit 4/3. STOP, no sweep. <<<")
    sys.exit(1)

print(f"{'(k,j)':>9} {'d':>2} {'2J':>3} {'dense?':>11} {'lam_max(M~)':>12} {'Q=Z(d2)':>9} "
      f"{'F(Tier2)':>9} {'F(n)':>9} {'plat?':>6} {'sat?':>6} {'agr':>4} {'mark':>18}")
rows = []
nonplateau = []
jlabel = {1: "1/2", 2: "1", 3: "3/2", 4: "2"}
for k, twoj, which in GRID:
    twoJ, d = (D.smallest_nontrivial(k, twoj) if which == "klein" else D.largest(k, twoj))
    gen = E.braid_generators(k, twoj, twoJ)
    if gen is None or gen["d"] < 2:
        continue
    d = gen["d"]; s1, s2 = gen["s1"], gen["s2"]
    arr = np.stack(E.herm_basis(d))
    # (1) density (hash enumeration)
    G = enumerate_group_fast(s1, s2, max_size=DICHTE_MAXSIZE)
    dense = G is None
    order = None if dense else len(G)
    # (2) tier-1 (max-over-Q). Q=Z diagnostic computed separately at full depth (d=2 only).
    lam, bestB, topB, _ = outer_max_tier1(s1, s2, arr, G, qz_Z=None)
    qz = qz_deep_d2(s1, s2, G, SEED + k) if d == 2 else None
    # (3) tier-2 only where tier-1 >= 3/2 - tol + convergence self-check F(n) vs F(3n)
    if lam >= LUEDERS - TOL:
        # B-set: small finite group -> ALL B (exact outer max); otherwise top-40 by lambda_max
        Bset = G if (G is not None and len(G) <= 500) else topB
        nr = NREST.get(d, 6000)
        s2seed = SEED + 100*k + twoj
        F_n = tier2_hard(Bset, arr, d, nr, s2seed)
        F = tier2_hard(Bset, arr, d, 3*nr, s2seed)
        conv_delta = F - F_n
        plateau = bool(conv_delta < CONV_TOL)
    else:
        F = None; F_n = None; conv_delta = None; plateau = None
    saturates = (F is not None and F >= LUEDERS - TOL)
    agree = (dense == saturates)
    mark = ""
    if dense and not saturates:
        mark = "dense_lam<3/2" if lam < LUEDERS - TOL else "dense_F<3/2,lam>=3/2"
    row = dict(k=k, j=jlabel[twoj], twoj=twoj, twoJ=twoJ, d=d,
               dense=bool(dense), group_order=order, lambda_max_Mtilde=lam,
               qz_K3=(None if qz is None else float(qz)),
               F_tier2=(None if F is None else float(F)),
               F_tier2_n=(None if F_n is None else float(F_n)),
               conv_delta=(None if conv_delta is None else float(conv_delta)), plateau=plateau,
               n_restart=(None if F is None else nr), Bset_size=(None if F is None else len(Bset)),
               saturates=bool(saturates), agree=bool(agree), mark=mark,
               luders_dist_lam=float(LUEDERS - lam))
    rows.append(row)
    if plateau is False:
        nonplateau.append((k, jlabel[twoj], conv_delta))
    print(f"{'('+str(k)+','+jlabel[twoj]+')':>9} {d:>2} {twoJ:>3} "
          f"{('dense' if dense else f'fin|G|={order}'):>11} {lam:>12.6f} "
          f"{('-' if qz is None else f'{qz:.6f}'):>9} "
          f"{('-' if F is None else f'{F:.6f}'):>9} "
          f"{('-' if F_n is None else f'{F_n:.6f}'):>9} "
          f"{('-' if plateau is None else ('Y' if plateau else 'N')):>6} "
          f"{('Y' if saturates else 'n'):>6} "
          f"{('OK' if agree else 'x'):>4} {mark:>18}")

# ---------- j=1/2 row: d=2-reference self-check (sampling adequacy, both readings) ----------
print("\n=== j=1/2 row: d=2-reference self-check (sampling adequacy; both readings) ===")
half = {r['k']: r for r in rows if r['twoj'] == 1}
checks = []
def near(a, b, t):
    return (a is not None) and abs(a - b) < t
# finite anchors, exact (enumerated):
checks.append(("k=4 max-Q=1.0", near(half.get(4, {}).get('lambda_max_Mtilde'), 1.0, 1e-6)))
checks.append(("k=4 Q=Z=1.0", near(half.get(4, {}).get('qz_K3'), 1.0, 1e-6)))
checks.append(("k=8 max-Q=1.4270509831", near(half.get(8, {}).get('lambda_max_Mtilde'), 1.4270509831, 1e-6)))
checks.append((f"k=8 Q=Z=3/sqrt5={SQRT3_5:.10f}", near(half.get(8, {}).get('qz_K3'), SQRT3_5, 1e-6)))
# dense saturation (sampling reaches the plateau ~1.4999 in BOTH readings):
EPS_SAT = 1e-3
for kk in (3, 5, 6, 7):
    r = half.get(kk, {})
    lam_ok = (r.get('lambda_max_Mtilde') is not None) and (1.5 - r['lambda_max_Mtilde'] < EPS_SAT)
    qz_ok = (r.get('qz_K3') is not None) and (1.5 - r['qz_K3'] < EPS_SAT)
    checks.append((f"k={kk} dense max-Q->approx1.4999 (dist {1.5-r.get('lambda_max_Mtilde',0):.2e})", lam_ok))
    checks.append((f"k={kk} dense Q=Z ->approx1.4999 (dist {1.5-r.get('qz_K3',0):.2e})", qz_ok))
d2_ref_ok = all(c[1] for c in checks)
for name, passed in checks:
    print(f"  [{'OK' if passed else 'X '}] {name}")
print(f"  -> self-check: {'PASS - j=1/2 row reproduces the d=2 reference in both readings, sampling adequate.' if d2_ref_ok else 'FAIL - sampling depth NOT adequate (dense j=1/2 below approx1.4999) -> increase depth.'}")

# ---------- tier-2 convergence overview (guard 1: plateau F(n) vs F(3n)) ----------
print("\n=== tier-2 convergence (guard 1: F(n) vs F(3n) -> plateau; guard 2: gate-0 4/3) ===")
print(f"  tier-2 gate-0 (4,1) d=3 -> 4/3: {'PASS' if T2G0 else 'FAIL'} (F={t2g0_F:.6f})")
if nonplateau:
    print(f"  NO plateau in {len(nonplateau)} cell(s): " +
          ", ".join(f"(k={kk},j={jj}) d={dd:.2e}" for kk, jj, dd in nonplateau) +
          " -> F not yet converged (more restarts needed).")
else:
    print(f"  all tier-2 cells reach a plateau (delta<{CONV_TOL}): F converged.")
tier2_converged = (len(nonplateau) == 0) and bool(T2G0)
print(f"  -> tier-2 solid: {'YES - gate-0 PASS + all plateaus.' if tier2_converged else 'NO - see above.'}")

print("\n>>> correspondence table complete. <<<")
results = dict(
    task="(k,j) density-witness sweep (correspondence table)",
    method="projective F/R B3-rep on d-dim fusion space; witness value=max_B lambda_max(Mtilde) [max-over-Q]; tier1/tier2",
    method_ref="method specification GATE-1 G1.0-G1.10",
    method_notes=dict(
        hash_speedup="enumerate_group_fast O(n^2)->O(n); cross-verified vs E.enumerate_group on k=4,k=8 (identical group+K3+lambda)",
        qz_column="Q=Z diagnostic (d=2 only) via K3_Qfixed, d=2 reference protocol; not the reported quantity; certifies j=1/2 row vs the d=2 reference",
        tier2_hardening="convergence fix (same quantity max-over-dichotomic-Q): eigen-seed from Mtilde top-eigvec ->+/-1 "
                        "+ broader top-B (40) + more restarts; guard1=plateau F(n) vs F(3n); guard2=gate-0 reproduces exact (4,1)->4/3"),
    ab_verify_pass=bool(AB),
    d2_ref_selfcheck_pass=bool(d2_ref_ok),
    tier2_gate0_pass=bool(T2G0), tier2_gate0_F=float(t2g0_F),
    tier2_converged=bool(tier2_converged),
    tier2_nonplateau=[dict(k=kk, j=jj, conv_delta=float(dd)) for kk, jj, dd in nonplateau],
    params=dict(tol=TOL, luders=LUEDERS, dichte_maxsize=DICHTE_MAXSIZE, sample_Lex=L_EX,
                n_rand=N_RAND, rand_lens=list(RAND_LENS), seed=SEED,
                tier2_nrest=NREST, conv_tol=CONV_TOL),
    engine_validated="gate-0 d=2 reproduces the d=2 reference exactly; Yang-Baxter ~1e-15 d=2..5 (p6_gate0_validate_results.json)",
    rows=rows,
    note="dense? via hash-enumeration to bound; reported quantity=max-over-Q (lambda_max_Mtilde); "
         "Q=Z column = d=2 reference reading (d=2 only); F=tier-2 dichotomic-Q lower bound",
)
with open(os.path.join(OUT, "p6_gate1_sweep_results.json"), "w") as fh:
    json.dump(results, fh, indent=2)
print(f"-> wrote p6_gate1_sweep_results.json")
