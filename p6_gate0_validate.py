"""
Gate-0 + engine validation (hard stop condition). Build check.

Checks BEFORE the sweep:
(1) gate-0 (d=2, j=1/2, own F/R engine): does it reproduce the d=2 reference EXACTLY?
    Q=Z: k=4 -> 1.000000, k=8 -> 3/sqrt5 = 1.341640786; max-Q: k=8 -> 1.427050983; the k=8 group closes at 120.
(2) engine sanity for j>=1 (d=3,4,5): F orthogonal, sigma1/sigma2 unitary, Yang-Baxter ~1e-15, (sigma1 sigma2)^3 scalar.
If (1) or (2) fails -> stop, engine error. No sweep.
"""
import numpy as np
import json
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import p6_engine as E
import p6_dkj_fusion_dims as D   # sector indices taken from D, not hardcoded

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
TOL = 1e-12


def sanity(gen):
    s1, s2 = gen["s1"], gen["s2"]
    d = gen["d"]
    Id = np.eye(d, dtype=complex)
    F = gen["F"]
    F_orth = float(np.max(np.abs(F @ F.T - np.eye(d))))
    u1 = float(np.max(np.abs(s1 @ s1.conj().T - Id)))
    u2 = float(np.max(np.abs(s2 @ s2.conj().T - Id)))
    yb = float(np.max(np.abs(s1@s2@s1 - s2@s1@s2)))
    c3 = np.linalg.matrix_power(s1@s2, 3)
    scal = float(np.max(np.abs(c3 - c3[0, 0]*Id)))
    return dict(F_orth=F_orth, unit1=u1, unit2=u2, yang_baxter=yb, cube_scalar=scal,
                pass_=bool(F_orth < TOL and u1 < TOL and u2 < TOL and yb < TOL and scal < TOL))


print("=== gate-0 + engine validation (build check) ===\n")
report = {}

# ---------- (1) gate-0: d=2, j=1/2, own F/R engine vs the d=2 reference ----------
print("(1) gate-0 -- own F/R engine at j=1/2 (d=2) vs the established d=2 reference values:")
basis2 = E.herm_basis(2)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
gate0 = {}
for k in (4, 8):
    gen = E.braid_generators(k, twoj=1, twoJ=1)   # j=1/2, smallest sector J=1/2, d=2
    san = sanity(gen)
    G = E.enumerate_group(gen["s1"], gen["s2"])
    if G is None:
        order = None
        maxQ = K3Z = None
        print(f"  k={k}: group does NOT close (>max_size) -- unexpected for j=1/2, k in {{4,8}}!")
    else:
        order = len(G)
        maxQ = max(E.Mtilde_lambda_max(B, basis2) for B in G)         # max-over-Q (bound)
        K3Z = max(E.K3_Qfixed(B, SZ) for B in G)                       # Q=Z fixed
    gate0[k] = dict(group_order=order, max_over_Q=maxQ, Q_Z=K3Z, sanity=san)
    print(f"  k={k}: |G|={order} | Q=Z K3={K3Z if K3Z is None else round(K3Z,9)} | "
          f"max-Q={maxQ if maxQ is None else round(maxQ,9)} | YB {san['yang_baxter']:.1e} F-orth {san['F_orth']:.1e}")

ref = dict(k4_QZ=1.0, k8_QZ=3/np.sqrt(5), k4_maxQ=1.0, k8_maxQ=1+2*np.cos(np.radians(72))-2*np.cos(np.radians(72))**2, k8_order=120)
print(f"\n  reference: Q=Z k=4 -> 1.0 k=8 -> {ref['k8_QZ']:.9f} | max-Q k=4 -> 1.0 k=8 -> {ref['k8_maxQ']:.9f} | k=8 |G| -> 120")
gate0_pass = (
    gate0[4]['Q_Z'] is not None and abs(gate0[4]['Q_Z']-1.0) < 1e-6 and
    abs(gate0[8]['Q_Z']-ref['k8_QZ']) < 1e-6 and
    abs(gate0[4]['max_over_Q']-1.0) < 1e-6 and
    abs(gate0[8]['max_over_Q']-ref['k8_maxQ']) < 1e-6 and
    gate0[8]['group_order'] == 120 and
    gate0[4]['sanity']['pass_'] and gate0[8]['sanity']['pass_']
)
print(f"  -> gate-0: {'PASS' if gate0_pass else 'FAIL -> stop, engine error'}")

# ---------- (2) engine sanity j>=1 (d=3,4,5) ----------
print("\n(2) engine sanity j>=1 (F orthogonal, sigma unitary, Yang-Baxter, (sigma1 sigma2)^3 scalar):")
# sector indices taken from D, not hardcoded (klein=smallest_nontrivial, gross=largest).
# covers exactly the dimensions the sweep actually computes: j=1->d=3, j=3/2->d=4, j=2->d=5.
cases = [("j=1,k=4   (klein)", 4, 2, "klein"), ("j=1,k=6   (klein)", 6, 2, "klein"),
         ("j=3/2,k=6 (gross)", 6, 3, "gross"), ("j=3/2,k=8 (gross)", 8, 3, "gross"),
         ("j=2,k=8   (gross)", 8, 4, "gross"), ("j=2,k=10  (gross)", 10, 4, "gross")]
sanity_all = {}
for name, k, twoj, which in cases:
    twoJ, _dexp = (D.smallest_nontrivial(k, twoj) if which == "klein" else D.largest(k, twoj))
    gen = E.braid_generators(k, twoj, twoJ) if twoJ is not None else None
    if gen is None or gen["d"] < 2:
        # no silently-filtered skip: d<2 despite a D sector = a real FAIL (sector/grid error).
        print(f"  {name}: 2J={twoJ} -> d<2 (no braiding sector) -- FAIL (check sector/grid!)")
        sanity_all[name] = dict(twoJ=twoJ, d=(None if gen is None else gen["d"]), pass_=False, no_run=True)
        continue
    san = sanity(gen)
    sanity_all[name] = dict(twoJ=twoJ, d=gen["d"], **san)
    print(f"  {name}: 2J={twoJ} d={gen['d']} | F-orth {san['F_orth']:.1e} | u1 {san['unit1']:.1e} u2 {san['unit2']:.1e} | "
          f"YB {san['yang_baxter']:.1e} | (sigma1 sigma2)^3 {san['cube_scalar']:.1e} -> {'PASS' if san['pass_'] else 'FAIL'}")
# skip/no_run counts as FAIL; additionally the sanity MUST actually cover d=3,4,5.
ds_covered = sorted({v["d"] for v in sanity_all.values() if v.get("d")})
dims_ok = {3, 4, 5}.issubset(set(ds_covered))
sanity_pass = bool(sanity_all) and all(v.get("pass_", False) for v in sanity_all.values()) and dims_ok
print(f"  [dim coverage of the sanity: d={ds_covered}  (must be superset of {{3,4,5}}): {'OK' if dims_ok else 'MISSING -> FAIL'}]")

print("\n=== validation status ===")
print(f"  gate-0 (d=2 reproduces the reference): {'PASS' if gate0_pass else 'FAIL'}")
print(f"  engine sanity j>=1 (YB etc.):          {'PASS' if sanity_pass else 'FAIL'}")
ok = gate0_pass and sanity_pass
print(f"  >>> {'engine validated -- sweep cleared.' if ok else 'stop -- engine error, no sweep.'} <<<")

report = dict(task="gate-0 + engine validation (build check)",
              gate0=gate0, gate0_reference=ref, gate0_pass=bool(gate0_pass),
              sanity_j_ge_1=sanity_all, sanity_pass=bool(sanity_pass),
              dim_coverage=ds_covered, dims_ge3_4_5_covered=bool(dims_ok),
              engine_validated=bool(ok),
              fix_note="sector indices taken from D instead of hardcoded (previously 2J=9/12 -> d<2 -> silently skipped); "
                       "a skip now counts as FAIL and d in {3,4,5} is enforced. Closes the false PASS.")
with open(os.path.join(OUT, "p6_gate0_validate_results.json"), "w") as fh:
    json.dump(report, fh, indent=2, default=lambda o: None)
print(f"\n-> wrote p6_gate0_validate_results.json")
