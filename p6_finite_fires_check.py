"""
Independent cross-check: does the TARGETED witness fire for FINITE reps?

For "fires <=> dense" the targeted witness (true LGI, rho-opt) must NOT fire (< 3/2) for FINITE
reps. Observation: finite (4,1) d=3 -> 1.5 (fires) -> the correspondence breaks at d>=3.
This is computed independently (own G_eff, own Q search via B-eigenvector pairs).

Targeted witness: K3(rho)=Tr[rho.G_eff], G_eff=D_Q(B^dag Q B)+B^dag D_Q(B^dag Q B) B-D_Q(B^2^dag Q B^2),
D_Q(X)=Pi+ X Pi+ - Pi- X Pi-.
rho-opt = lambda_max(sym(G_eff)).  (Neutral witness rho=I/d = (1/d)Tr[sym(G_eff)] as a sanity check.)

Mechanism Q-seed: in B's eigenbasis B acts on each eigenvector pair (i,j) as a rotation by
(phi_i - phi_j). Q=+/-1 on the pair -> reduces to the d=2 value at that angle. Angle=pi/3 -> 3/2.
Test ALL pairs + sign choices on the rest + random Q. Finite -> enumerate.
"""
import numpy as np
import json
import os
import sys
import itertools
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import scipy.linalg as sla
import p6_engine as E
import p6_dkj_fusion_dims as D

HERE = os.path.dirname(os.path.abspath(__file__))


def DQ(X, Q):
    d = len(Q); Pp = (np.eye(d) + Q)/2; Pm = (np.eye(d) - Q)/2
    return Pp @ X @ Pp - Pm @ X @ Pm


def G_eff(B, Q):
    Bd = B.conj().T; B2 = B @ B; B2d = B2.conj().T
    A = Bd @ Q @ B
    return DQ(A, Q) + Bd @ DQ(A, Q) @ B - DQ(B2d @ Q @ B2, Q)


def targeted(B, Q):   # ρ-opt
    G = G_eff(B, Q); return float(np.linalg.eigvalsh((G + G.conj().T)/2)[-1])


def neutral(B, Q, d):  # ρ=I/d
    G = G_eff(B, Q); return float(np.real(np.trace((G + G.conj().T)/2))/d)


def eig_pair_Qs(B):
    """Q-seeds from B-eigenvector pairs via Schur (orthonormal basis Z -> Q exactly dichotomic):
    Q = Z diag(+/-1) Z^dag, Q^2=I guaranteed. +1 at i, -1 at j, +/- on the rest (tests the pi/3 mechanism)."""
    d = B.shape[0]
    T, Z = sla.schur(B, output='complex')   # B = Z T Z^dag, Z unitary (for normal B: T diagonal)
    out = []
    for i, j in itertools.combinations(range(d), 2):
        rest = [r for r in range(d) if r not in (i, j)]
        for signs in itertools.product([1, -1], repeat=len(rest)):
            diag = np.zeros(d); diag[i] = 1; diag[j] = -1
            for r, s in zip(rest, signs):
                diag[r] = s
            Q = Z @ np.diag(diag.astype(complex)) @ Z.conj().T   # Z unitary -> Q exactly dichotomic
            out.append(Q)
    return out


def rand_dich(d, rng):
    r = int(rng.integers(1, d)); M = rng.normal(size=(d, r)) + 1j*rng.normal(size=(d, r))
    Vv, _ = np.linalg.qr(M); return 2*(Vv @ Vv.conj().T) - np.eye(d)


def cell_max(k, twoj, which, n_rand=60, seed=0):
    twoJ, d = (D.largest(k, twoj) if which == "gross" else D.smallest_nontrivial(k, twoj))
    gen = E.braid_generators(k, twoj, twoJ); d = gen["d"]
    G = E.enumerate_group(gen["s1"], gen["s2"], max_size=4000)
    dense = G is None
    if dense:   # for comparison only: sample
        gens = np.stack([E.to_sud(gen["s1"]), E.to_sud(gen["s1"].conj().T), E.to_sud(gen["s2"]), E.to_sud(gen["s2"].conj().T)])
        rng = np.random.default_rng(seed); G = []
        for _ in range(1500):
            idx = rng.integers(0, 4, size=12); B = np.eye(d, dtype=complex)
            for t in idx: B = B @ gens[t]
            G.append(B)
    rng = np.random.default_rng(seed+5)
    best_t = -9.0; best_n = -9.0
    for B in G:
        Qs = eig_pair_Qs(B) + [rand_dich(d, rng) for _ in range(n_rand)]
        for Q in Qs:
            best_t = max(best_t, targeted(B, Q))
            best_n = max(best_n, neutral(B, Q, d))
    return d, dense, best_t, best_n


if __name__ == "__main__":
    print("=== cross-check: does the targeted witness fire for FINITE reps? ===")
    print("    'fires' := targeted-rho-opt MAX >= 1.49. correspondence needs: finite -> do NOT fire.\n")
    print(f"{'(k,j)':>9} {'d':>2} {'type':>8} {'neutral(I/d)':>13} {'targeted(rho-opt)':>17} {'fires?':>9}")
    cells = [(4, 1, "klein"), (8, 1, "klein"), (4, 2, "klein"), (5, 2, "klein"), (8, 4, "gross")]
    jl = {1: "1/2", 2: "1", 4: "2"}
    out = []
    for k, tj, wh in cells:
        d, dense, bt, bn = cell_max(k, tj, wh, seed=20260602+k)
        fires = bt >= 1.49
        print(f"{'('+str(k)+','+jl[tj]+')':>9} {d:>2} {('dense' if dense else 'finite'):>8} "
              f"{bn:>13.5f} {bt:>17.5f} {('Y' if fires else 'n'):>9}")
        out.append(dict(k=k, j=jl[tj], d=d, dense=bool(dense), neutral_max=bn, targeted_max=bt, fires=bool(fires)))
    # conclusion
    fin_fire = [(r["k"], r["j"], r["d"]) for r in out if (not r["dense"]) and r["fires"]]
    print(f"\n  FINITE reps that FIRE (targeted >=1.49): {fin_fire}")
    print(f"  -> if finite d>=3 reps fire: 'fires <=> dense' is broken for the targeted witness at d>=3.")
    with open(os.path.join(HERE, "p6_finite_fires_check.json"), "w") as fh:
        json.dump(dict(note="cross-check: does targeted witness (G_eff rho-opt) fire for FINITE reps? "
                            "correspondence needs finite->NOT fire. fires:=targeted>=1.49",
                       cells=out, finite_firing=fin_fire), fh, indent=2)
    print("\n>>> cross-check complete. <<<")
