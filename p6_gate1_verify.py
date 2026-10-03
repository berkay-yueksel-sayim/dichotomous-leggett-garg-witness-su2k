#!/usr/bin/env python3
"""
Verification layer for the (k,j) density-witness correspondence table.

This is a validation layer, not the discovery search. The reported optima in
p6_gate1_sweep_results.json are given as explicit braid words; the search
infrastructure that produced them is not part of this record (see Data and Code
Availability). Given a braid word, however, recomputing lambda_max(Mtilde) is a
closed-form, deterministic calculation, and for the finite representations the
full group can be enumerated exactly.

Two independent checks per row:

  (1) closed form       -- rebuild sigma_1, sigma_2 from p6_engine.braid_generators,
                           evaluate the deposited braid word, recompute
                           lambda_max(Mtilde) and compare against the stored value.
  (2) full enumeration  -- for rows with dense == false the braid image is finite;
                           the group is enumerated in full and the maximum taken
                           over every element. This is the stronger check: it does
                           not rely on the deposited word being optimal, it proves
                           it.

Neutral vs. targeted reading: the table stores the targeted (state-optimized)
quantity max_Q lambda_max(Mtilde). The neutral (rho = I/d) values are closed form
and live in p6_rho_closed.json: (3d-1)/(2d) for odd d, 3/2 for even d.

Outputs
  p6_gate1_verify_results.json  -- per-row verdicts
  p6_gate1_resolution_map.png   -- the two-panel resolution map (see README)

Run:  python p6_gate1_verify.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p6_engine as E  # noqa: E402

TOL = 1e-6
SWEEP = os.path.join(HERE, "p6_gate1_sweep_results.json")
RHO = os.path.join(HERE, "p6_rho_closed.json")
FIN = os.path.join(HERE, "p6_finite_fires_check.json")
OUT = os.path.join(HERE, "p6_gate1_verify_results.json")
MAP = os.path.join(HERE, "p6_gate1_resolution_map.png")

# levels swept in this record; beyond that the paper states the scope explicitly
K_RELEASED_MAX = 10
K_INTERNAL_MAX = 16


def _load(path):
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _rows(doc):
    """Row list, whatever the top-level key is called."""
    if isinstance(doc, list):
        return doc
    if "rows" in doc:
        return doc["rows"]
    for value in doc.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    return []


def word_to_matrix(word, s1, s2):
    """Evaluate a braid word over the two generators. '1'/'2' and inverses '1i'/'2i'."""
    dim = s1.shape[0]
    out = np.eye(dim, dtype=complex)
    i = 0
    while i < len(word):
        ch = word[i]
        inv = (i + 1 < len(word) and word[i + 1] in "iI'")
        gen = s1 if ch == "1" else s2 if ch == "2" else None
        if gen is None:
            raise ValueError("unexpected symbol %r in braid word" % ch)
        out = out @ (np.conjugate(gen).T if inv else gen)
        i += 2 if inv else 1
    return out


def check_row(row, basis_cache):
    """One row -> verdict dict. Never adjusts a value; only reports."""
    k, twoj, twoJ = row["k"], row["twoj"], row["twoJ"]
    d, stored = row["d"], row["lambda_max_Mtilde"]
    verdict = {"k": k, "j": row["j"], "d": d, "dense": row["dense"],
               "stored_lambda_max_Mtilde": stored}

    gen = E.braid_generators(k, twoj, twoJ)
    if gen is None:
        verdict["closed_form"] = {"status": "sector space has dimension < 2"}
        verdict["full_enumeration"] = {"status": "not applicable"}
        return verdict
    s1, s2 = gen["s1"], gen["s2"]
    if gen["d"] != d:
        verdict["dimension_mismatch"] = {"engine": gen["d"], "stored": d}
    if d not in basis_cache:
        basis_cache[d] = E.herm_basis(d)
    basis = basis_cache[d]

    # --- (1) closed form from the deposited word ---------------------------
    word = row.get("argmax_word")
    if word:
        try:
            B = word_to_matrix(word, s1, s2)
            got = float(E.Mtilde_lambda_max(B, basis))
            verdict["closed_form"] = {"word": word, "value": got,
                                      "delta": got - stored,
                                      "match": abs(got - stored) <= TOL}
        except Exception as exc:                       # noqa: BLE001
            verdict["closed_form"] = {"word": word, "error": repr(exc)}
    else:
        verdict["closed_form"] = {"status": "no argmax_word in this row"}

    # --- (2) full group enumeration, finite rows only ----------------------
    if not row["dense"]:
        group = E.enumerate_group(s1, s2)
        best = max(float(E.Mtilde_lambda_max(g, basis)) for g in group)
        # No argmax pointer is reported. For a finite representation the claim is
        # "no group element exceeds this value" -- that is established by the
        # enumeration itself, not by exhibiting a witness. An index would only be
        # meaningful relative to the enumeration order, which is not part of this
        # record, and would break silently the next time that order changes.
        verdict["full_enumeration"] = {
            "group_order": len(group),
            "group_order_stored": row.get("group_order"),
            "group_order_match": row.get("group_order") in (None, len(group)),
            "max": best, "delta": best - stored,
            "match": abs(best - stored) <= TOL}
        # The two row types carry DIFFERENT claims and must not be reported alike.
        # finite: universal  -- "no group element exceeds this"  -> exhaustion settles it
        # dense:  existential -- "this value is attained by this word" -> a witness IS the evidence
        verdict["claim"] = "universal"
        verdict["evidence"] = ("settled by full enumeration over |G| = %d" % len(group)
                               if verdict["full_enumeration"]["match"]
                               else "full enumeration DISAGREES")
    else:
        verdict["full_enumeration"] = {
            "status": "representation is dense; full enumeration not applicable"}
        verdict["claim"] = "existential"
        cf = verdict["closed_form"]
        # Deliberately NOT "maximum verified". For a dense (infinite) image the true
        # supremum is not attained and cannot be certified by recomputation; the paper
        # itself says so ("supremum ... never exactly attained", "sampling estimates").
        # What this check establishes is bookkeeping: the deposited word belongs to the
        # reported value. The checking output must not claim more than the paper does.
        verdict["evidence"] = (
            "witness recomputed: deposited word reproduces the reported value"
            if cf.get("match") else
            "witness DISAGREES: deposited word does not reproduce the reported value"
            if cf.get("match") is False else
            "no witness deposited yet (argmax_word missing)")
    return verdict


# ---------------------------------------------------------------- the map
def neutral_value(d, rho_rows):
    """Neutral (rho = I/d) reading, closed form, read from p6_rho_closed.json."""
    for r in rho_rows:
        if r["d"] == d:
            return float(r["K3_rho_Id"])
    return float("nan")


def build_map(sweep_rows, rho_rows, path):
    """Two panels, same axes and scale: neutral (left) and targeted (right).

    Every cell is a deposited number. The panels are shown side by side because
    the two readings are different quantities -- putting the protocol in a
    caption instead of on the axis is exactly the confusion this figure exists
    to prevent.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:                            # noqa: BLE001
        return {"status": "matplotlib unavailable", "error": repr(exc)}

    js = ["1/2", "1", "3/2", "2"]
    ks = sorted({r["k"] for r in sweep_rows})
    by = {(r["k"], r["j"]): r for r in sweep_rows}
    d_of = {r["j"]: r["d"] for r in sweep_rows}

    neu = np.full((len(js), len(ks)), np.nan)
    tar = np.full((len(js), len(ks)), np.nan)
    for a, j in enumerate(js):
        for b, k in enumerate(ks):
            row = by.get((k, j))
            if row is None:
                continue
            tar[a, b] = row["lambda_max_Mtilde"]
            # at d = 2 the neutral and targeted readings coincide (state-independent),
            # so the per-k structure is the neutral structure; for d >= 3 the neutral
            # value is the closed form and does not depend on k
            neu[a, b] = row["lambda_max_Mtilde"] if row["d"] == 2 \
                else neutral_value(row["d"], rho_rows)

    # Row verdict for the neutral reading. This is deliberately NOT encoded in the
    # color scale: "open" is a state, not a value -- giving it a color from the
    # value scale would turn it into a fourth number.
    #   separates      d = 2      finite reps sit below the cap, dense reps on it
    #   no separation  odd d >= 3 dense and finite alike reach the cap
    #   open           even d >= 4  the paper leaves this case open
    def verdict(dd):
        if dd == 2:
            return "separates", "#1a7f37"
        if dd % 2 == 1:
            # Line break, not a shorter word: rotated 90 deg the label has to
            # fit INSIDE one row height (~0.55 in). "no separation" needs
            # ~0.63 in at 7 pt; broken, the longest line "separation" needs
            # ~0.49 in. Same words.
            return "no\nseparation", "#b3261e"
        return "open", "#8a8a8a"

    n_extra = 3                      # placeholder columns for k > released max
    # Column-spanning figure*: the target width is \textwidth = 510.00 pt
    # = 7.0569 in, measured from the record's own documentclass line. Side
    # by side, two panels of 13.1 x-units with five-digit numbers inside the
    # cells do not fit that width, so the panels are stacked. No sharey:
    # stacked, both panels are leftmost and both carry their row labels.
    fig, axes = plt.subplots(2, 1, figsize=(7.0569, 8.60),
                             constrained_layout=True)
    vmin, vmax = 1.0, 1.5
    # The two panels are two DIFFERENT quantities, not one quantity under two
    # settings. Labelling both with lambda_max(Mtilde) would name the left panel
    # after the right one's observable; they coincide only at d = 2.
    titles = ["neutral reading  ($\\rho = I/d$)\n"
              "$K_3(\\rho=I/d)$   —   $\\lambda_{\\max}(\\tilde M)$ only at $d=2$",
              "targeted reading  ($\\rho$-optimized)\n"
              "$\\max_Q \\lambda_{\\max}(\\tilde M)$"]
    for ax, data, title in zip(axes, (neu, tar), titles):
        im = ax.imshow(data, aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
                       cmap="viridis")
        ax.set_xlim(-1.6, len(ks) - .5 + n_extra)
        ax.set_xticks(list(range(len(ks))) + [len(ks) - .5 + n_extra / 2.0])
        ax.set_xticklabels([str(k) for k in ks] + ["$k\\leq%d$" % K_INTERNAL_MAX])
        ax.set_yticks(range(len(js)))
        ax.set_yticklabels(["$j=%s$\n$d=%d$ (%s)"
                            % (j, d_of[j], "even" if d_of[j] % 2 == 0 else "odd")
                            for j in js])
        ax.set_xlabel("level $k$")
        ax.set_title(title, pad=26)

        for a, j in enumerate(js):
            for b, k in enumerate(ks):
                row = by.get((k, j))
                if row is None:                       # sector space has dim < 2
                    ax.add_patch(Rectangle((b - .5, a - .5), 1, 1,
                                           facecolor="white", edgecolor="0.8", lw=.6))
                    ax.text(b, a, "n/a", ha="center", va="center",
                            fontsize=7.0, color="0.55")
                    continue
                if not row["dense"]:
                    ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False,
                                           lw=2.4, edgecolor="white"))
                    # $|G|{=}n$ instead of $|G|=n$: mathtext puts wide binary-
                    # operator space around "=", which made the label wider than
                    # its cell (0.50 in of text in a 0.375 in cell). Same
                    # characters, tight binding -- the idiom this file already
                    # uses for "$k{=}3$ saturates" below.
                    # Two lines, not a smaller font (7 pt is the floor): a
                    # three-digit order needs 0.42 in on one line and the cell
                    # is 0.424 in wide, so |G|=162 and |G|=120 spilled into the
                    # neighbouring cell -- where white text on a white n/a cell
                    # simply vanishes. Every character is kept, including the
                    # "=", which moves to the second line with its value.
                    ax.text(b, a - .28, "$|G|$\n$=%d$" % row["group_order"],
                            ha="center", va="center", fontsize=7.0, color="white")
                ax.text(b, a + .08, "%.3f" % data[a, b], ha="center", va="center",
                        fontsize=8,
                        color="white" if data[a, b] < 1.42 else "black")

            # scope band: levels checked internally but not released
            ax.add_patch(Rectangle((len(ks) - .5, a - .5), n_extra, 1,
                                   facecolor="0.93", edgecolor="0.75",
                                   hatch="///", lw=.6))

        ax.text(len(ks) - .5 + n_extra / 2.0, (len(js) - 1) / 2.0,
                "verified internally, $k\\leq%d$;\nnot part of this record"
                % K_INTERNAL_MAX,
                ha="center", va="center", rotation=90, fontsize=7.4, color="0.35")

        # verdict gutter, left of the grid -- neutral panel only
        if data is neu:
            for a, j in enumerate(js):
                label, color = verdict(d_of[j])
                ax.add_patch(Rectangle((-1.5, a - .5), .8, 1, facecolor=color,
                                       edgecolor="white", lw=.8, alpha=.9))
                ax.text(-1.1, a, label, ha="center", va="center", rotation=90,
                        fontsize=7.0, color="white", fontweight="bold")
            ax.text(-1.1, len(js) - .5 + .10, "does the neutral\nreading separate?",
                    ha="center", va="bottom", fontsize=7, color="0.3")

    # the k = 4 comparison -- the single most counter-intuitive cell in the table.
    # Placed BELOW the grid: an annotation that covers a verified cell is worse
    # than none (the first pass hid |G|=162).
    ax0 = axes[0]
    # The k=3/k=4 annotation below the grid lives in data coordinates with
    # clip_on=False. constrained_layout does not account for artists outside
    # the axes, so with stacked panels it would land in the lower panel's
    # title. Extending this panel's y-range downward puts the annotation
    # inside the axes box, where the layout engine can see it. The imshow
    # data still spans -0.5..3.5; the added space is empty.
    ax0.set_ylim(-1.75, len(js) - .5)
    i3, i4 = ks.index(3), ks.index(4)
    for i in (i3, i4):
        ax0.annotate("", xy=(i, -.5), xytext=(i, -.95),
                     annotation_clip=False,
                     arrowprops=dict(arrowstyle="->", color="0.25", lw=1.1))
    ax0.plot([i3, i4], [-.95, -.95], color="0.25", lw=1.1, clip_on=False)
    # Broken over two lines and centred on the panel instead of on the k=3/k=4
    # midpoint: at \textwidth the single line is 7.4 in wide and ran off the
    # figure. The arrows above still mark the two columns, so the sentence
    # does not have to sit under them. Wording unchanged.
    ax0.text((len(ks) - 1) / 2.0, -1.02,
             "$k{=}3$ saturates, $k{=}4$ is inert ($K_3{=}1$ on every axis)\n"
             "— despite $k{=}4$ having the larger quantum dimension",
             ha="center", va="top", fontsize=7.6, color="0.15", clip_on=False)

    fig.colorbar(im, ax=axes, label="witness value  $K_3$",
                 fraction=.022, pad=.02)
    # The parity rule belongs ON the panel, not above it: what must not fall off
    # does not belong in a caption. Placed in the empty n/a block, upper left.
    ax0.text(2.6, len(js) - 1, "neutral cap\n"
             "$(3d-1)/(2d)$  for odd $d$\n$3/2$  for even $d$",
             ha="center", va="center", fontsize=8.4, color="0.2",
             bbox=dict(boxstyle="round,pad=.4", fc="white", ec="0.6", lw=.8))
    # suptitle, not a free fig.text: constrained_layout reserves space for a
    # suptitle but not for figure-coordinate text, so the line landed inside
    # the first panel title. Same words, same size.
    fig.suptitle("every cell is a deposited value",
                 fontsize=8, color="0.35")
    # No bbox_inches: with it the saved width is the CONTENT extent and stops
    # following figsize (measured here: figsize 15.4 in wrote a 14.165 in
    # file). With constrained_layout and no bbox_inches the output width
    # equals figsize, so the print scale in a figure* is 1.0 by construction
    # and the fontsize in this script IS the printed point size.
    # Metadata is suppressed here rather than stripped afterwards.
    fig.savefig(path, dpi=200, metadata={"Software": None})
    fig.savefig(os.path.splitext(path)[0] + ".pdf",
                metadata={"Creator": None, "Producer": None})
    plt.close(fig)
    return {"status": "written", "file": os.path.basename(path),
            "panels": ["neutral", "targeted"], "cells": int(np.isfinite(tar).sum())}


def main():
    sweep = _load(SWEEP)
    rows = _rows(sweep)
    rho_rows = _rows(_load(RHO))

    basis_cache = {}
    verdicts = [check_row(r, basis_cache) for r in rows]

    checked = [v for v in verdicts
               if v["closed_form"].get("match") is not None
               or v["full_enumeration"].get("match") is not None]
    failed = [v for v in verdicts
              if v["closed_form"].get("match") is False
              or v["full_enumeration"].get("match") is False]

    uni = [v for v in verdicts if v.get("claim") == "universal"]
    exi = [v for v in verdicts if v.get("claim") == "existential"]
    print("rows in table                    : %d" % len(rows))
    print("universal claims (finite reps)   : %d settled by full enumeration, %d open"
          % (sum(1 for v in uni if v["full_enumeration"].get("match")),
             sum(1 for v in uni if not v["full_enumeration"].get("match"))))
    print("existential claims (dense reps)  : %d witness recomputed, %d without witness"
          % (sum(1 for v in exi if v["closed_form"].get("match")),
             sum(1 for v in exi if v["closed_form"].get("match") is None)))
    print("rows with a verdict              : %d" % len(checked))
    print("MISMATCHES                       : %d" % len(failed))
    for v in failed:
        print("  k=%s j=%s d=%s stored=%.10f  closed=%s  enum=%s"
              % (v["k"], v["j"], v["d"], v["stored_lambda_max_Mtilde"],
                 v["closed_form"].get("value"), v["full_enumeration"].get("max")))

    fig_info = build_map(rows, rho_rows, MAP)
    print("resolution map           : %s" % fig_info.get("status"))

    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"task": "verification of the (k,j) density-witness table",
                   "tolerance": TOL,
                   "rows": verdicts,
                   "mismatches": len(failed),
                   "figure": fig_info}, fh, indent=2)
    print("written                  : %s" % os.path.basename(OUT))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
