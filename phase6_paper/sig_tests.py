#!/usr/bin/env python3
"""Significance tests for the reach and lithology contrasts in Table 11.

County-level means are the unit of analysis (n = 1,068), because the 36,210
county-window pairs are not independent within a county.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

R = "/projects/sandbox/Assesment_of_Ecological_resilience_Yangtze/"
n1 = pd.read_csv(R + "Outputs_v3/outputs_v3/PERSIST/interpretability/n1_coefficients.csv")
n1["recovery"] = 1 - n1.rho
n1["resistance"] = 1 - n1.kap.abs()
cty = n1.groupby("adcode")[["rho", "sig", "kap", "recovery", "resistance"]].mean()
cty.index = cty.index.astype(str)
rch = n1.groupby("adcode").reach.first()
rch.index = rch.index.astype(str)
cty["reach"] = rch

p = pd.read_parquet(R + "phase3_data/tables/panel_monthly.parquet")
p["adcode"] = p.adcode.astype(str)
st = p.groupby("adcode")[["karst_frac", "elev_mean", "relief", "slope_mean",
                          "ntl_mean"]].first()
cty = cty.join(st, how="inner")
cty["kcls"] = pd.cut(cty.karst_frac, [-.01, .001, .5, 1.01],
                     labels=["Non-karst", "Partially karst", "Majority karst"])
print(f"n counties = {len(cty)}")

OUT = {}


def eps2(H, n, k):
    """Epsilon-squared effect size for Kruskal-Wallis."""
    return float((H - k + 1) / (n - k))


def omnibus(group_col, var, label):
    groups = [g[var].values for _, g in cty.groupby(group_col, observed=True)]
    names = [str(k) for k, _ in cty.groupby(group_col, observed=True)]
    H, pv = stats.kruskal(*groups)
    e2 = eps2(H, len(cty), len(groups))
    F, pf = stats.f_oneway(*groups)
    print(f"\n{label} :: {var}")
    print(f"  Kruskal-Wallis H({len(groups)-1}) = {H:.2f}, p = {pv:.3e}, "
          f"epsilon^2 = {e2:.3f}")
    print(f"  one-way ANOVA F({len(groups)-1},{len(cty)-len(groups)}) = {F:.2f}, "
          f"p = {pf:.3e}")
    for nm, g in zip(names, groups):
        print(f"    {nm:<18} n={len(g):>4}  mean={g.mean():.4f}  sd={g.std():.4f}")
    # pairwise Mann-Whitney with Holm correction
    pairs, raw = [], []
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            u, pu = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
            # Cliff's delta from U
            na, nb = len(groups[a]), len(groups[b])
            delta = 2 * u / (na * nb) - 1
            pairs.append((names[a], names[b], delta))
            raw.append(pu)
    order = np.argsort(raw)
    adj = np.empty(len(raw))
    run = 0.0
    for rank, idx in enumerate(order):
        val = raw[idx] * (len(raw) - rank)
        run = max(run, val)
        adj[idx] = min(run, 1.0)
    for (a, b, dl), pu, pa in zip(pairs, raw, adj):
        star = "***" if pa < .001 else "**" if pa < .01 else "*" if pa < .05 else "ns"
        print(f"    {a} vs {b}: Cliff d = {dl:+.3f}, p_raw = {pu:.2e}, "
              f"p_holm = {pa:.2e} {star}")
    OUT[f"{label}|{var}"] = dict(H=H, p=pv, eps2=e2, F=F, pF=pf,
                                 pairs=[(a, b, float(d), float(pa))
                                        for (a, b, d), pa in zip(pairs, adj)])


print("=" * 74)
print("OMNIBUS AND PAIRWISE TESTS")
print("=" * 74)
for var in ["resistance", "recovery", "sig"]:
    omnibus("reach", var, "Reach")
for var in ["resistance", "recovery", "sig"]:
    omnibus("kcls", var, "Lithology")

print("\n" + "=" * 74)
print("SPEARMAN CORRELATIONS WITH PHYSIOGRAPHY (county means, n = 1,068)")
print("=" * 74)
rows = []
for v in ["karst_frac", "elev_mean", "relief", "slope_mean", "ntl_mean"]:
    for tgt in ["recovery", "resistance", "sig"]:
        rp, pp = stats.pearsonr(cty[v], cty[tgt])
        rs, ps = stats.spearmanr(cty[v], cty[tgt])
        rows.append(dict(driver=v, target=tgt, pearson_r=rp, pearson_p=pp,
                         spearman_rho=rs, spearman_p=ps))
        print(f"  {v:<12} vs {tgt:<11} Pearson r={rp:+.3f} (p={pp:.2e})  "
              f"Spearman rho={rs:+.3f} (p={ps:.2e})")
pd.DataFrame(rows).to_csv("/projects/sandbox/yreb_resilience/sig_correlations.csv",
                          index=False)
with open("/projects/sandbox/yreb_resilience/sig_tests.json", "w") as f:
    json.dump(OUT, f, indent=2, default=float)
print("\nsaved sig_tests.json + sig_correlations.csv")
