#!/usr/bin/env python3
"""Resolving power of the Delta screen rule (docs/delta/rules.md, seed count and advance rule).

Design arithmetic, not a result. Reads delta.json for the family sizes m: the (entry, target) cells in a
wave x target that get a G3 accuracy test. A cell is counted from its role field:
  - tier "baseline" never counts (the four baselines are comparands, reported without an advance decision);
  - at 350k, a cell counts when it is an accuracy screen on d24h2, a floor-family cell (accuracy against
    d24h2 by Welch) or an FF2 cell; the feasibility probes do not;
  - M010 (1.4M) is a ladder point and never counts.
alpha_eff = q / m is the threshold of the first Benjamini-Hochberg discovery (conservative: the j-th uses j*q/m).

Two k values per (n, alpha), each such that the stated rule has 80 % power when the true effect is k * sd_d:
  k_t     the one-sided paired t test alone (gate (i)); noncentral t, exact;
  k_joint the whole advance rule, (i) and (ii) mean >= g_0/2, with g_0 = k_joint * sd_d the minimum
          detectable effect. Monte Carlo, 200,000 reps, fixed seed.

Usage: python3 screen_power.py [dir-with-delta.json]    (needs numpy and scipy)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
A = json.loads((D / "delta.json").read_text())
Q, POWER, THRESH, FLOOR_FRAC, REPS = 0.10, 0.80, 0.3, 0.5, 200_000
NS = (4, 6, 8)
ACCURACY_ROLES = {"near-floor-d24h2", "own-arch", "d24h2-base"}


def k_t(se_f, df, alpha, power=POWER):
    """Effect / sigma at which a one-sided t test (SE = se_f * sigma, df) has the given power."""
    tc = stats.t.ppf(1 - alpha, df)
    return optimize.brentq(lambda d: (1 - stats.nct.cdf(tc, df, d / se_f)) - power, 1e-6, 1e3)


def _draws(df, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(REPS), np.sqrt(rng.chisquare(df, REPS) / df)


def joint_power(k, se_f, df, alpha, floor_frac, draws):
    """P(t > t_c and effect_hat >= floor_frac * k) when the true effect is k (sigma = 1)."""
    z, s = draws
    eff = k + se_f * z
    tc = stats.t.ppf(1 - alpha, df)
    return float(np.mean((eff / (s * se_f) > tc) & (eff >= floor_frac * k)))


def k_joint(se_f, df, alpha, floor_frac=FLOOR_FRAC, power=POWER, seed=20260927):
    draws = _draws(df, seed)
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if joint_power(mid, se_f, df, alpha, floor_frac, draws) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def paired(n):
    return 1 / math.sqrt(n), n - 1


def is_accuracy_cell(e, target):
    if e["tier"] == "baseline" or target not in e["targets"]:
        return False
    if target == 350000:
        return e.get("role_350k") in ACCURACY_ROLES
    return True


def m_of(entries, wave, target, tiers):
    return sum(1 for e in entries if e["wave"] == wave and e["tier"] in tiers and is_accuracy_cell(e, target))


E = A["entries"]
fam = {
    "W2 5M (singles)": m_of(E, "W2", 5000000, ("single",)),
    "W2 350k (singles)": m_of(E, "W2", 350000, ("single",)),
    "W3 5M (packages)": m_of(E, "W3", 5000000, ("package",)),
    "W3 350k (packages)": m_of(E, "W3", 350000, ("package",)),
}
probes = [e["id"] for e in E if e.get("role_350k") == "near-floor-d24h2-probe"]
bl = [e["id"] for e in E if e["tier"] == "baseline"]
print("BH families (wave x target cells with a G3 accuracy test; baselines", ", ".join(bl),
      "and the 350k probes", ", ".join(probes), "excluded):")
for k_, m in fam.items():
    print(f"  {k_}: m = {m}, alpha_eff = q/m = {Q / m:.5f}")

print(f"\nPaired screen cells, sd_d units. k_t: gate (i) alone. k_joint: (i) and (ii) mean >= {FLOOR_FRAC} * g_0,"
      f" 80 % power at g_0 = k_joint * sd_d. ceiling = {THRESH} / k_joint (pt).")
rows = [("unadjusted alpha 0.05", 0.05)] + [(k_, Q / m) for k_, m in fam.items()]
for label, a in rows:
    parts = []
    for n in NS:
        se, df = paired(n)
        kt, kj = k_t(se, df, a), k_joint(se, df, a)
        parts.append(f"n={n}: k_t={kt:.2f} k_joint={kj:.2f} ceiling={THRESH / kj:.3f}")
    print(f"  {label:24s} " + "  ".join(parts))

print(f"\nFactorials (per-run sigma; BH within the block's effects; n seeds per cell; (ii) floor {FLOOR_FRAC} * g_0):")
for name, cells, params_fixed, m in (("FF1", 16, 15, 15), ("FF2", 8, 7, 7)):
    for n in NS:
        runs = cells * n
        df = runs - (n + params_fixed)          # intercept + (n - 1) seed blocks + effects
        se = math.sqrt(4 / runs)                # effect = mean(high) - mean(low), runs/2 each
        kt, kj = k_t(se, df, Q / m), k_joint(se, df, Q / m)
        print(f"  {name} n={n}: {runs} runs, df {df}, m {m}: k_t = {kt:.2f}, k_joint = {kj:.2f}, sigma ceiling = {THRESH / kj:.3f}")

a = Q / fam["W2 5M (singles)"]
se8, df8 = paired(8)
kj8 = k_joint(se8, df8, a)
print("\nIllustrative g_res = k_joint(8, alpha_eff of W2 5M) * sd_plan, with sd_plan = sqrt(2) * sd (illustrative sd values, not measurements):")
for sd in (0.1, 0.3, 0.6, 1.0):
    sp = math.sqrt(2) * sd
    print(f"  sd {sd:.1f} pt: sd_plan {sp:.2f} pt -> g_res(8) = {kj8 * sp:.2f} pt")
print(f"  largest epoch-500 sd for n = 8 at W2 5M: {THRESH / kj8 / math.sqrt(2):.3f} pt")
se3, df3 = paired(3)
print(f"\nCheap version (3 seeds): k_joint(3) = {k_joint(se3, df3, a):.2f} at the W2 5M alpha_eff, {k_joint(se3, df3, 0.05):.2f} unadjusted")
