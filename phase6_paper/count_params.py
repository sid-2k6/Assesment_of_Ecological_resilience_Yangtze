#!/usr/bin/env python3
"""Exact per-module parameter counts for PERSIST v3 as configured in the run.

Extracts the PERSIST/GraphAttn classes from the v3 notebook so the numbers in
the paper's architecture tables cannot drift from the code that produced the
results.
"""
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NB = ("/projects/sandbox/Assesment_of_Ecological_resilience_Yangtze/"
      "phase5_proposed_model/PERSIST_Phase5_v3.ipynb")

nb = json.load(open(NB))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

# config exactly as run
D_MODEL, N_EXPERTS, GRAPH_HEADS, DROPOUT = 96, 4, 2, 0.20
GRAPH_SCALE_INIT, RES_SCALE_INIT, WARM_START_N1 = 0.30, 0.30, True
F_DYN, F_STAT, F_ANN, N_OUT = 32, 8, 64, 2
IDX_STATE = list(range(5))    # lst_c, kndvi, ndvi_mean, lst_ds, kndvi_ds
IDX_FORCE = list(range(22))   # 21 TerraClimate vars + n_rel

g = dict(torch=torch, nn=nn, F=F, math=math,
         D_MODEL=D_MODEL, N_EXPERTS=N_EXPERTS, GRAPH_HEADS=GRAPH_HEADS,
         DROPOUT=DROPOUT, GRAPH_SCALE_INIT=GRAPH_SCALE_INIT,
         RES_SCALE_INIT=RES_SCALE_INIT, WARM_START_N1=WARM_START_N1,
         IDX_STATE=IDX_STATE, IDX_FORCE=IDX_FORCE)

start = src.index("class GraphAttn")
end = src.index('print("PERSIST v3 defined')
exec(compile(src[start:end], "<persist>", "exec"), g)

model = g["PERSIST"](F_DYN, F_STAT, F_ANN, N_OUT)
total = sum(p.numel() for p in model.parameters())
print(f"TOTAL PARAMETERS = {total:,}   (run log reported 356,081)")
assert total == 356081, f"mismatch: {total}"
print()

GROUPS = {
    "Monthly temporal encoder (N5)":   ["month_gru", "month_norm"],
    "Annual temporal encoder (N5)":    ["year_gru", "year_norm", "a_year"],
    "Forcing encoder":                 ["force_enc"],
    "State encoder":                   ["state_enc", "a_state"],
    "Spatial graph attention (N2b)":   ["g_sp", "norm_sp", "a_sp"],
    "Hydrological graph attention (N2a)": ["g_hy", "norm_hy", "a_hy"],
    "Static context encoder":          ["static_enc"],
    "Lithology expert gate (N3)":      ["gate"],
    "Attribution head (N3/N6)":        ["attr"],
    "Coefficient experts (N1, 4x)":    ["coef"],
    "Forcing-effect head (N1)":        ["f_head"],
    "Free residual head (R1)":         ["direct", "a_res"],
    "Reconstruction head (N4)":        ["recon"],
}

named = dict(model.named_parameters())
seen, rows = set(), []
for label, prefixes in GROUPS.items():
    n = 0
    for name, p in named.items():
        if any(name == pre or name.startswith(pre + ".") for pre in prefixes):
            n += p.numel(); seen.add(name)
    rows.append((label, n))

leftover = {k: v.numel() for k, v in named.items() if k not in seen}
print(f"{'Component':<40} {'Parameters':>12}  {'Share':>7}")
print("-" * 63)
for label, n in rows:
    print(f"{label:<40} {n:>12,}  {100*n/total:>6.2f}%")
print("-" * 63)
print(f"{'TOTAL':<40} {sum(n for _, n in rows):>12,}  "
      f"{100*sum(n for _, n in rows)/total:>6.2f}%")
if leftover:
    print("\nUNGROUPED:", leftover)

print("\n=== layer-wise detail ===")
LAYERS = [
    ("Dynamic input window", "32 features x 12 months", "(B, 12, N, 32)"),
    ("Annual input window", "64 features x 3 years", "(B, 3, N, 64)"),
    ("Static input", "8 physiographic features", "(N, 8)"),
    ("Monthly GRU (N5)", f"GRU(32 -> 96), 2 layers, dropout {DROPOUT}", "(B, N, 96)"),
    ("Monthly LayerNorm", "LayerNorm(96)", "(B, N, 96)"),
    ("Annual GRU (N5)", "GRU(64 -> 96), 1 layer", "(B, N, 96)"),
    ("Annual scaled residual", "a_year * LayerNorm(96), a_year init 0.10", "(B, N, 96)"),
    ("Forcing encoder", "Linear(22 -> 96) + ELU + Linear(96 -> 96)", "(B, N, 96)"),
    ("State encoder", "Linear(5 -> 96) + ELU + Linear(96 -> 96)", "(B, N, 96)"),
    ("Spatial graph attention (N2b)", "2-head masked attention, 3,032 edges", "(B, N, 96)"),
    ("Hydro graph attention (N2a)", "2-head masked attention, 3,032 directed edges", "(B, N, 96)"),
    ("Gated graph fusion", "a_sp, a_hy init 0.30 (learnable)", "(B, N, 96)"),
    ("Static context encoder", "Linear(8 -> 96) + ELU", "(N, 96)"),
    ("Concatenation", "[h ; context]", "(B, N, 192)"),
    ("Attribution head (N6)", "Linear(192 -> 4) + softmax stream gating", "(B, N, 4)"),
    ("Lithology expert gate (N3)", "Linear(8 -> 4) + softmax", "(N, 4)"),
    ("Coefficient experts (N1)", "4 x [Linear(192 -> 96) + ELU + Dropout + Linear(96 -> 8)]", "(B, N, 4, 8)"),
    ("Expert mixing", "gate-weighted sum over 4 experts", "(B, N, 8)"),
    ("Bounded coefficients (N1)", "sigmoid(rho), sigmoid(sigma), tanh(kappa), delta", "4 x (B, N, 2)"),
    ("Forcing-effect head (N1)", "Linear(192 -> 96) + ELU + Linear(96 -> 2)", "(B, N, 2)"),
    ("Free residual head (R1)", "Linear(192 -> 96) + ELU + Dropout + Linear(96 -> 2), a_res 0.30", "(B, N, 2)"),
    ("Reconstruction head (N4)", "Linear(96 -> 32)", "(B, N, 32)"),
    ("Output", "rho*y_prev + sigma*y_seas + kappa*f + delta + a_res*direct", "(B, N, 2)"),
]
for a, b, c in LAYERS:
    print(f"  {a:<32} | {b:<58} | {c}")
