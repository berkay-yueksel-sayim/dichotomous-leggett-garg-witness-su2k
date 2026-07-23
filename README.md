# Paper 4 v1.1 — Zenodo Build

**Title:** A dichotomous Leggett–Garg witness for braid-representation density in SU(2)_k: exact at d=2, dimension-limited at d≥3
**Author:** Berkay Yuksel Sayim
**Email:** berksa@tutamail.com
**ORCID:** [0009-0004-4993-7352](https://orcid.org/0009-0004-4993-7352)
**Affiliation:** Independent Research, Germany
**Version:** 1.1
**Date:** 2026-07-17
**Resource type:** Preprint
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## v1.1 changes (this release)

- Attribution fix: the {2,4,8} finite-image classification is now attributed to
  Freedman--Larsen--Wang (the k=8 icosahedral borderline case via Kuperberg), not to
  Kuperberg alone; Tuba--Wenzl and Rowell--Tuba are added as the underlying B₃-finiteness
  references, at the three load-bearing sites.
- The {2,4,8} finite set is qualified throughout (abstract + 8 body/appendix sites) as
  specific to the minimal-rank, one-qubit ($d=2$) fusion space, distinct from the
  all-rank finite set $k\in\{1,2,4\}$.
- **Math correction:** the closed-form passage stating "$\hat z$ is a 5-fold axis at
  $k=8$ ($\sigma_1=144^\circ$)" is corrected. $\sigma_1$ (144° about the measurement
  axis $\hat z$) gives $K_3=1$, not $3/\sqrt5$; the value $3/\sqrt5$ is realized by a
  distinct 72° rotation about a *different* tilted 5-fold axis. Fixed in both Sec. III.B
  and Appendix C. No numerical value changes (3/√5 was and remains correct); only the
  angle/element attribution is corrected.
- Added the geometric discriminator distinguishing binary (SU(2), |G|=48/24/120) from
  projective (SO(3), order 24/12/60) group orders, and the general sector-phase family
  $\delta^\ast(k)=\pi k/(k+2)$ unifying the σ₁ angles at k=2,4,8.
- The $d\ge5$ scope statement is sharpened from "implied, not verified" to structural:
  no finite $d=5$ anyon representation exists to test for $k\le16$ (verified via an
  internal level sweep, $k\le16$, not included in this deposit; the in-deposit sweep
  artifacts here cover $k\le10$, see `p6_gate1_sweep_results.json`).
- A companion-work citation to the Fibonacci-only ($k=3$) Leggett--Garg Letter
  (Concept-DOI 10.5281/zenodo.20372744) is added in the Introduction.
- A one-sentence limitation on weak/nonprojective measurement protocols is added to
  the Discussion.
- `\setcounter{secnumdepth}{3}` added (series convention); PDF-metadata subject field
  added; date placeholder set (filled at release).
- No numerical result, figure, or existing claim beyond the angle/element attribution
  above is changed.

## Additional corrections (internal review pass, same v1.1 release)

A subsequent fresh-context review pass found one further blocker and several smaller
items inherited from v1.0/v1.1; corrected here, no result/figure/table value changes.

- **B-4 (blocker): "at every $d\ge2$" contradicted the paper's own Table I and its own
  later sentence.** Sec.~III.D said the state-optimized value reaches $3/2$ "for any
  nontrivial dynamics at every $d\ge2$", but Table I and the very next subsection
  ("State-dependence appears only at $d\ge3$": at $d=2$ neutral and targeted readings
  *coincide*, at $k=4\to1.0/1.0$ and $k=8\to1.427/1.427$ — neither is $3/2$) both show
  the targeted witness does *not* reach $3/2$ at $d=2$. Corrected to "$d\ge3$" (matches
  the section's own title, "Loss of resolution at $d\ge3$"). The identical wording in
  Appendix C ("the state-optimized value is $3/2$ for every $d\ge2$") is corrected the
  same way; "value" is also corrected to "cap" there for terminology consistency with
  the paragraph heading ("Caps.").
- **4-F1:** the bare "$k=8$ is pinned at $3/\sqrt5$" (Abstract + Conclusion + this
  README) lacked the $Q=\hat z$-axis qualifier stated correctly in the body (Table I,
  Sec.~III.B, App. C); axis-optimized $k=8$ is $1.427$, not $3/\sqrt5$. Qualifier
  "(at $Q=\hat z$; $1.427$ axis-optimized)" added at both top-level sites.
- **4-F2:** "the underlying $B_3$ finiteness established by Tuba--Wenzl and
  Rowell--Tuba" overstated TW/RT's role relative to Freedman--Larsen--Wang (who
  identify the $\{2,4,8\}$ finite set). Reworded to "the underlying $B_3$
  representation-theoretic machinery due to Tuba--Wenzl (classification of $B_3$
  representations) and Rowell--Tuba (finite-image criterion)".
- **4-F3:** the README claimed the $k\le16$ no-finite-$d=5$-representation scope
  statement was "verified against `d5_sweep.json`" — that file is not part of this
  deposit (it lives outside it, with a hardcoded local path, and was not sanitized for
  inclusion). Reworded to state the $k\le16$ check is an internal level sweep not
  included here, and to point to the actual in-deposit sweep artifacts
  (`p6_gate1_sweep_results.json`), which cover $k\le10$.
- **4-F4:** "has not been studied" (Abstract, this README) lacked the "to our
  knowledge" hedge used elsewhere in the series for priority claims. Added.
- **M-1:** the closed-form identity $f(c,1)=1$ (Sec.~III.B, App. C) is used without a
  justification; a half-sentence is added noting $2c-2c^2=2c(1-c)$ cancels the
  $-2pc(1-c)$ term exactly at $p=1$, for every $\theta$.
- **Not built (deferred):** the two build-backup files formerly kept alongside
  the source for audit-trail purposes were removed at the pre-upload scrub, per
  series convention. Remaining cosmetic minors are not built in this pass.

## Abstract

Whether a temporal (Leggett–Garg) measurement can certify the computational power of an
anyonic braid representation — its density in the unitary group, the property underlying
universal topological quantum computation — has, to our knowledge, not been studied.
We give a complete
operational characterization of when the dichotomous Lüders–Leggett–Garg witness
K₃ = 2 C(B) − C(B²) resolves braid-representation density across the SU(2)_k family.
At d=2 (the spin-½ fusion space) the witness resolves density exactly: it saturates the
dimension-independent Lüders bound 3/2 on every dense representation and stays strictly
below it on the finite ones, which occur precisely at k ∈ {2,4,8} — a sharp threshold.
The k=4 representation is structurally inert (K₃=1 for every measurement axis) despite a
larger quantum dimension than k=3, its braid image being finite; and k=8 is pinned at
3/√5 (at Q=ẑ; 1.427 axis-optimized). Both follow from the underlying SO(3) geometry.
At d≥3 the same witness ceases to
resolve density — it saturates 3/2 even on finite representations — and we trace this to
the extra dimension itself, not to any particular braid angle. We frame the resulting
no-go question — whether every such witness is density-blind at d≥3 — as an open problem.

## Related work

This preprint is part of a series on Bell and Leggett–Garg correlations in
Fibonacci-anyon and SU(2)_k braiding. It extends to the full SU(2)_k family the
temporal-inequality companion (the Leggett–Garg letter):

> B. Y. Sayim, *Leggett–Garg saturation and structural signatures in
> Fibonacci-anyon braiding*, Zenodo preprint (2026), Concept-DOI
> [10.5281/zenodo.20372744](https://doi.org/10.5281/zenodo.20372744).

Companion works in the same series:

> B. Y. Sayim, *Sector-Dependent CHSH Violation in Fibonacci Anyons, and
> Finite-Size Topological--CHSH Mutual Information in 2D Lattice Models*, Zenodo
> preprint (2026), Concept-DOI
> [10.5281/zenodo.19600752](https://doi.org/10.5281/zenodo.19600752).
>
> B. Y. Sayim, *Generic CHSH Violation in Fibonacci Anyon Braiding: A Landscape
> Analysis*, Zenodo preprint (2026), Concept-DOI
> [10.5281/zenodo.19601352](https://doi.org/10.5281/zenodo.19601352).
>
> B. Y. Sayim, *CHSH-Form Values Without Bell Nonlocality: Inapplicability of the
> Tsirelson Bound on a Non-Factorized Fibonacci Fusion Space*, Zenodo preprint
> (2026), Concept-DOI
> [10.5281/zenodo.19601998](https://doi.org/10.5281/zenodo.19601998).

*(Companion titles above reflect each paper's current build as of 2026-07-09;
Concept-DOIs always resolve to the latest version regardless of title changes.)*

---

## What's in this archive

**Paper**

| File | Role |
|---|---|
| `main_v1.1.tex` | LaTeX source (RevTeX 4-2, PRX style) |
| `main_v1.1.pdf` | Compiled preprint |
| `paper4_fig1_K3_vs_k_d2.pdf`, `.png` | Figure 1 |
| `paper4_fig1_K3_vs_k_d2.py` | Figure-1 builder (self-contained; data embedded) |
| `LICENSE` | CC BY 4.0 |
| `README.md` | This file |

**Reproduction code (NumPy + SciPy) and result data (JSON)**

| File | Role |
|---|---|
| `p6_engine.py` | projective F/R (q-6j/Racah) B_3 engine on the fusion space |
| `p6_gate0_validate.py` + `…_results.json` | gate-0 / Yang–Baxter engine validation (App. A) |
| `p6_dkj_fusion_dims.py` + `…_results.json` | fusion-space dimension per (k,j) and total-charge sector (App. B, Table `tab:sectors`) |
| `p6_gate1_sweep.py` + `…_results.json` | (k,j) density-witness sweep (Table I, Fig. 1 data) |
| `p6_rho_closed.py` + `p6_rho_closed.json` | closed-form caps (3d−1)/(2d), ρ-opt (§IV, App. C) |
| `p6_finite_fires_check.py` + `…check.json` | neutral vs targeted at finite d=3 (§IV) |
| `p6_airtight_finite_d3.py` + `…_results.json` | targeted witness fires on finite d=3 reps (Table II) |
| `p6_nail_dimension.py`, `p6_nail_independent.py` + `…_results.json` (each) | dimension nail, two independent solvers (Table III) |
| `derive_3sqrt5.py` + `derive_3sqrt5_results.json` | closed-form 3/√5 at k=8 (§III, App. C) |
| `p6_k4_lambda_max.py` + `gate1_lambda_max_results.json` | k=4 inertness, λ_max=1 for every axis (§III) |

## Key numerical claims (independently verifiable from the JSON result files)

| Claim | Value | Source field |
|---|---|---|
| d=2 sweep, axis-optimized K₃ (Fig. 1 / Table I) | k=2,4→1.000; k=8→1.427; dense k=3,5,6,7,9,10→1.4997–1.5000 | `p6_gate1_sweep_results.json` → `rows[].lambda_max_Mtilde` |
| k=8 at Q=ẑ | 3/√5 = 1.3416407865 (θ=72°, n_z²=1/5) | `derive_3sqrt5_results.json` → `global_max` |
| k=4 structurally inert | K₃=1 for every axis (max_U λ_max=1.0 over the 24-element group) | `gate1_lambda_max_results.json` |
| Finite d=3 reps fire (targeted) | (4,1)\|162, (8,2)\|450, (12,3)\|882 → all 1.500 | `p6_airtight_finite_d3_results.json` → `rows`; cf. `p6_finite_fires_check.json` |
| Closed caps | (3d−1)/(2d) odd d, 3/2 even d; ρ-opt=3/2 ∀d | `p6_rho_closed.json` (`all_match: true`) |
| Dimension nail (2 solvers agree) | d=2: 60°→1.5, 72°→1.427, 90/100°→1.0; d=3⊕triv: all 1.500 | `p6_nail_dimension_results.json`, `p6_nail_independent_results.json` |
| Fusion-sector dims (App. B, Table `tab:sectors`) | (4,1)→{0:1,2:3,4:1}; (8,2)→{0:1,2:3,4:5,6:3,8:1}; (12,3)→{0:1,2:3,4:5,6:7,8:5,10:3,12:1} | `p6_dkj_fusion_dims_results.json` → `table_sectors_cells` |
| Gate-0 / Yang–Baxter sanity | YB residual ≲ 10⁻¹⁵ for d=2..5 | `p6_gate0_validate_results.json` |

## Reproduction

Deterministic. NumPy and SciPy (plus Matplotlib for the figure); random seeds are fixed in the
scripts.

```bash
python p6_gate0_validate.py         # engine gate-0 / Yang-Baxter validation
python p6_gate1_sweep.py            # (k,j) density-witness sweep  -> Table I, Fig. 1 data
python p6_rho_closed.py             # closed-form caps (3d-1)/(2d), rho-opt
python p6_finite_fires_check.py     # neutral vs targeted at finite d=3
python p6_airtight_finite_d3.py     # targeted witness fires on finite d=3 reps  -> Table II
python p6_nail_dimension.py         # dimension nail (algebraic solver)  -> Table III
python p6_nail_independent.py       # dimension nail (independent solver)  -> Table III
python derive_3sqrt5.py             # closed-form 3/sqrt5 at k=8
python p6_k4_lambda_max.py          # k=4 inertness (lambda_max = 1 for every axis)
python paper4_fig1_K3_vs_k_d2.py    # Figure 1
```

Gate-0 and Yang–Baxter sanity checks at the head of the engine (`p6_engine.py`) must pass
at machine precision; if they do not, the F/R-symbol conventions have been altered.

## Build

- LaTeX engine: MiKTeX pdfTeX-1.40.29 (3 × pdflatex passes, converged; 0 overfull boxes,
  0 undefined references/citations).
- Bibliography: full-text verified (every title/volume/page/DOI). FLW = Commun. Math.
  Phys. **228** (not 227); MM page 2265 confirmed via Crossref; corroboration experiments
  Zhan et al. (PRA 107, 012424, 2023) and Tusun et al. (PRA 105, 042613, 2022).
- Python: 3.12, NumPy, SciPy, and Matplotlib.
