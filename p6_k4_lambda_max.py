"""
Closed-form max_U lambda_max(M(U)).

Question: for k=4, is there ANY axis m-hat with K3>1?  max over m-hat = lambda_max(M(U))
(Rayleigh / Courant-Fischer, closed form via eigenvalue -- not sampling). The condition is
max_U lambda_max <= 1 over all 24 elements (a true for-all). delta=0, braiding only, j=1/2 2D-rep.

Closed-form eigenstructure (analytic, checked against eigvalsh in the script):
  For a rotation by angle theta (c=cos theta), axis n-hat:  sym(R)=c I+(1-c) n n^T, sym(R^2)=cos2theta I+(1-cos2theta) n n^T
  => M(U)=2 sym(R)-sym(R^2) = (1+2c-2c^2) I - 2c(1-c) n n^T.
  EIGENVALUES: along n-hat -> 1 (exact);  perpendicular to n-hat (2-fold) -> 1+2c-2c^2.
  => lambda_max(M(U)) = max(1, 1+2c-2c^2).  1+2c-2c^2>1 only for c in (0,1), max 3/2 at c=1/2 (theta=60deg).
  So the max over the group depends ONLY on whether some rotation angle theta in (0deg,90deg) occurs.

Method: numpy eigvalsh on the symmetric M(U) (exact), per element, max over the 24/120.
"""
import numpy as np
import json
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SIG = [SX, SY, SZ]


def qint(n, k): return np.sin(n*np.pi/(k+2.0))/np.sin(np.pi/(k+2.0))
def to_su2(U): return U/np.sqrt(np.linalg.det(U))


def tl_gens(k):
    d = qint(2, k); A = 1j*np.exp(-1j*np.pi/(2.0*(k+2.0))); s = np.sqrt(d**2-1.0)
    e1 = np.array([[d, 0], [0, 0]], dtype=complex)
    e2 = np.array([[1/d, s/d], [s/d, d-1/d]], dtype=complex)
    s1, s2 = to_su2(A*I2+e1/A), to_su2(A*I2+e2/A)
    return [s1, s2, s1.conj().T, s2.conj().T]


def enumerate_group(gens, tol=1e-8, max_size=400):
    elems = [I2.copy()]
    def seen(U): return any(np.max(np.abs(U-V)) < tol for V in elems)
    frontier = [I2.copy()]
    for g in gens:
        if not seen(g): elems.append(g); frontier.append(g)
    while frontier:
        nf = []
        for U in frontier:
            for g in gens:
                W = to_su2(U@g)
                if not seen(W): elems.append(W); nf.append(W)
        frontier = nf
        if len(elems) > max_size: return None
    return elems


def R_so3(U):
    R = np.zeros((3, 3)); Ud = U.conj().T
    for i in range(3):
        for j in range(3):
            R[i, j] = 0.5*np.real(np.trace(SIG[i] @ U @ SIG[j] @ Ud))
    return R


def sym(A): return 0.5*(A + A.T)


def analyze(k, sphere_scan=False, n_scan=200000, seed=12345):
    G = enumerate_group(tl_gens(k))
    best = dict(lam=-9, theta_deg=None, axis=None, mhat=None)
    analytic_err = 0.0
    sym_err = 0.0
    for U in G:
        R = R_so3(U)
        M = 2*sym(R) - sym(R@R)
        sym_err = max(sym_err, np.max(np.abs(M - M.T)))
        w, V = np.linalg.eigh(M)          # symmetric -> exact real eigenvalues/vectors (closed form, no sampling)
        lam = float(w[-1]); mhat = V[:, -1]
        # analytic cross-check: theta from Tr(R)=1+2cos theta; eigenvalues should be {1, 1+2c-2c^2(x2)}
        c = np.clip((np.trace(R) - 1.0)/2.0, -1.0, 1.0)
        analytic = np.sort([1.0, 1+2*c-2*c**2, 1+2*c-2*c**2])
        analytic_err = max(analytic_err, np.max(np.abs(np.sort(w) - analytic)))
        if lam > best["lam"]:
            best = dict(lam=lam, theta_deg=float(np.degrees(np.arccos(c))),
                        c=float(c), mhat=[float(x) for x in mhat])
    res = dict(group_order=len(G), max_lambda_max=best["lam"], maximizer_theta_deg=best["theta_deg"],
               maximizer_mhat=best["mhat"], symmetry_err=float(sym_err),
               analytic_vs_eigvalsh_err=float(analytic_err))
    if sphere_scan:   # cross-check only
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(n_scan, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        scan_best = -9
        for U in G:
            R = R_so3(U); M = 2*sym(R) - sym(R@R)
            q = np.einsum('ni,ij,nj->n', v, M, v)   # m^T M m for all sampled m
            scan_best = max(scan_best, float(np.max(q)))
        res["sphere_scan_max"] = scan_best
        res["scan_consistent"] = bool(scan_best <= best["lam"] + 1e-6)
    return res


print("=== closed-form max_U lambda_max(M(U)) ===\n")
print("  method: eigvalsh (exact eigenvalues of the symmetric 3x3 M(U)), closed form. max over m-hat = lambda_max.")
print("  analytic: eig(M) = {1, 1+2c-2c^2, 1+2c-2c^2}, c=cos(theta_U).\n")

out = {}
# k=4 = the relevant case (sphere scan as a cross-check)
out[4] = analyze(4, sphere_scan=True)
# k=2 cross-sanity, k=8 contrast (FLW-universal)
out[2] = analyze(2)
out[8] = analyze(8)

for k in (4, 2, 8):
    r = out[k]
    tag = {4: "relevant case", 2: "cross-sanity (octahedron)", 8: "contrast (FLW-universal)"}[k]
    print(f"k={k} [{tag}] |G|={r['group_order']}:")
    print(f"   max_U lambda_max(M(U)) = {r['max_lambda_max']:.10f}")
    print(f"   maximizer: theta={r['maximizer_theta_deg']:.2f}deg, m-hat~[{', '.join(f'{x:.3f}' for x in r['maximizer_mhat'])}]")
    print(f"   M symmetric: {r['symmetry_err']:.1e} | analytic==eigvalsh: {r['analytic_vs_eigvalsh_err']:.1e}")
    if 'sphere_scan_max' in r:
        print(f"   [cross-check] sphere scan max = {r['sphere_scan_max']:.6f}  "
              f"consistent(<=lambda_max)? {r['scan_consistent']}")
    print()

print("=== result (closed-form for-all step) ===")
print(f"  k=4:  max_U lambda_max(M(U)) = {out[4]['max_lambda_max']:.10f}")
print(f"  k=2 (sanity):  {out[2]['max_lambda_max']:.10f}   |   k=8 (contrast): {out[8]['max_lambda_max']:.10f}")
print("  (k=4 rotation angles are only 120deg/180deg [+id] -> c in {-1/2,-1} -> 1+2c-2c^2<=1 -> lambda_max=1 per element.)")
print("  >>> done. <<<")

results = dict(
    gate="closed-form max_U lambda_max(M(U))",
    method="numpy eigvalsh on symmetric M(U)=2 sym(R)-sym(R^2); exact eigenvalues; max over finite group. NOT sampling.",
    analytic_eigenstructure="M(U)=(1+2c-2c^2) I - 2c(1-c) n n^T ; eigenvalues {1, 1+2c-2c^2(x2)}; c=cos(theta_U)",
    over_claim_flag="delta=0, braiding only, scope j=1/2 2D-rep",
    k4=out[4], k2=out[2], k8=out[8],
    note_k4="all k=4 rotation angles in {0,120,180} deg -> 1+2c-2c^2 <= 1 -> lambda_max=1 per element -> max=1",
    interpretation="the closed-form for-all step gives max_U lambda_max=1 for k=4; the value only, scoped to j=1/2 2D-rep",
)
with open(os.path.join(OUT, "gate1_lambda_max_results.json"), "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\n-> wrote gate1_lambda_max_results.json")
