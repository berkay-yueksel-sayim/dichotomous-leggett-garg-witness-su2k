"""
Fusion-space dimension d per (k, j).

Computes the SU(2)_k fusion-space dimension of 3 anyons of spin j per
total-charge sector -- the input quantity for the (k, j) grid of the
density-witness study.

Doubled labels: a = 2j (int), so half-integrality stays exact.
SU(2)_k truncation:  |a-b| <= c <= min(a+b, 2k-a-b),  c same parity as a+b.

Verified: j=1/2 -> d=2 (smallest non-trivial sector, all k>=2). Small cases
cross-checked against hand calculation.

Findings:
  - dim>2 requires j>=1 (for j=1/2 every non-trivial sector has d=2).
  - first genuine dim>2 case: (k>=4, j=1), d=3.
  - higher d via the largest instead of the smallest sector: j=3/2 -> d=4 (k>=6), j=2 -> d=5 (k>=8).
  - j ceiling from the M-bound: none (d^2 <= 25 for d<=5; the bottleneck is braid enumeration).
"""


def allowed(a, b, c, k):
    """Does SU(2)_k fusion allow j1 x j2 -> j3 ?  (a=2j1, b=2j2, c=2j3, k level)."""
    if (a + b + c) % 2:           # parity
        return False
    if c < abs(a - b) or c > a + b:
        return False
    if c > 2 * k - a - b:         # level-k truncation
        return False
    if c < 0 or c > 2 * k:
        return False
    return True


def dims(k, twoj):
    """d(J) per total-charge sector 2J for 3 anyons with label twoj=2j.
    Fusion path ((j x j)->i) x j -> J:  d(J) = sum_i N(j,j,i) * N(i,j,J)."""
    res = {}
    for tJ in range(0, 2 * k + 1):            # 2J
        d = 0
        for ti in range(0, 2 * k + 1):        # 2 * intermediate charge
            if allowed(twoj, twoj, ti, k) and allowed(ti, twoj, tJ, k):
                d += 1
        if d > 0:
            res[tJ] = d
    return res


def smallest_nontrivial(k, twoj):
    """Smallest total-charge sector with d>=2 (otherwise braiding has no effect)."""
    d = dims(k, twoj)
    for tJ in sorted(d):
        if d[tJ] >= 2:
            return tJ, d[tJ]
    return None, None


def largest(k, twoj):
    """Largest-d sector (for deliberately higher d in the grid)."""
    d = dims(k, twoj)
    if not d:
        return None, None
    tJ = max(d, key=lambda t: d[t])
    return tJ, d[tJ]


if __name__ == "__main__":
    import json
    from pathlib import Path

    OUT = Path(__file__).parent
    jlabel = {1: "1/2", 2: "1", 3: "3/2", 4: "2"}
    print("k    j     sectors(2J:d)                          smallest-nontriv (2J,d)    largest (2J,d)     d^2")
    print("-" * 108)
    rows = []
    for twoj in [1, 2, 3, 4]:
        for k in range(2, 13):
            if twoj > 2 * k:                  # j <= k/2
                continue
            d = dims(k, twoj)
            tJs, dds = smallest_nontrivial(k, twoj)
            tJl, ddl = largest(k, twoj)
            sect = " ".join(f"{t}:{d[t]}" for t in sorted(d))
            nt = f"(2J={tJs}, d={dds})" if tJs is not None else "all d=1 (no braiding)"
            lg = f"(2J={tJl}, d={ddl})"
            d2 = dds * dds if dds else "-"
            rows.append({"k": k, "j": jlabel[twoj], "sectors": d,
                         "smallest_nontrivial": {"twoJ": tJs, "d": dds} if tJs is not None else None,
                         "largest": {"twoJ": tJl, "d": ddl}})
            print(f"{k:<4} {jlabel[twoj]:<5} {sect:<38} {nt:<26} {lg:<18} {d2}")
        print()

    # Direct check of the three (k, j) cells behind Table tab:sectors in the main
    # text (App. B): (4,1), (8,2), (12,3), each read with the doubled label
    # twoj = 2*j directly (outside the twoj in [1,2,3,4] sweep range above, which
    # does not reach j=3 / twoj=6).
    print("=== Table tab:sectors cells (App. B) ===")
    table_sectors_check = []
    for k, j, twoj in [(4, 1, 2), (8, 2, 4), (12, 3, 6)]:
        d = dims(k, twoj)
        table_sectors_check.append({"k": k, "j": j, "sectors": d})
        print(f"  (k={k}, j={j}): {d}")
    (OUT / "p6_dkj_fusion_dims_results.json").write_text(json.dumps(
        {"meta": {"script": "p6_dkj_fusion_dims.py"}, "rows": rows,
         "table_sectors_cells": table_sectors_check}, indent=2))
    print(f"WRITE {OUT / 'p6_dkj_fusion_dims_results.json'}")
