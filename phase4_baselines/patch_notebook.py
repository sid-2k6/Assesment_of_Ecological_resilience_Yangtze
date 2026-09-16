#!/usr/bin/env python3
"""Patch build_notebook.py with the Phase-4 fixes:

1. TRIVIAL BASELINES (climatology / county-climatology / persistence /
   seasonal-naive) so there is finally a floor to compare against.
2. Single GLOBAL scale for the primary target instead of per-county-month
   division, which was producing 61-sigma outliers that dominated every
   squared-error metric.
3. Primary target changed to DESEASONALISED state (regional month climatology
   removed, county structure RETAINED) so spatial signal survives and the graph
   is actually testable. The strict county-demeaned anomaly is still reported,
   via an exact affine back-transform, as the hard secondary view.
4. Trend handled with a year feature + expanding-climatology option.
5. Variance decomposition diagnostic (between- vs within-county) so the source
   of any R2 is transparent.
"""
import re
import pathlib

SRC = pathlib.Path("build_notebook.py")
s = SRC.read_text()


def replace_block(text, start_marker, end_marker, new, label):
    """Replace text between two markers (inclusive of neither)."""
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    print(f"  patched: {label}")
    return text[:i] + new + text[j:]


# ─────────────────────────────────────────────── 1. CONFIG additions
old_cfg = '''# --- task -------------------------------------------------------------
LOOKBACK   = 12           # months of history fed to the model
HORIZON    = 1            # forecast 1 month ahead
TARGETS    = ["lst_z", "kndvi_z"]      # response channels (LST is N1 primary)
'''
new_cfg = '''# --- task -------------------------------------------------------------
LOOKBACK   = 12           # months of history fed to the model
HORIZON    = 1            # forecast 1 month ahead

# PRIMARY targets: deseasonalised state. The regional month climatology is
# removed (so the trivial seasonal cycle cannot be exploited) but county-level
# structure is RETAINED, which is what makes the graph testable and the task
# learnable at all. Scaled by ONE global sd per target.
TARGETS    = ["lst_ds", "kndvi_ds"]

# STRICT targets: county x month demeaned anomalies - the hard interannual
# signal. Reported via an exact affine back-transform, never trained on.
STRICT_TARGETS = ["lst_z", "kndvi_z"]

# target hygiene (the previous run produced a 61-sigma kNDVI value because a
# near-zero county sd was used as a divisor)
MIN_SD_FRAC = 0.10        # county sd floored at this fraction of the global sd
CLIP_SIGMA  = 5.0         # hard clip on standardised targets
'''
s = s.replace(old_cfg, new_cfg)
print("  patched: CONFIG targets + hygiene")

# ─────────────────────────────────────────────── 2. target construction
old_start = '''# ---- train-only climatology -> leakage-free anomalies ----------------
train_mask = panel.year.between(*TRAIN_YEARS)'''
old_end = '''# ---- feature groups --------------------------------------------------'''

new_targets = '''# ---- train-only climatologies -> leakage-free targets -----------------
train_mask = panel.year.between(*TRAIN_YEARS)
RAW_OF = {"lst_ds": "lst_c", "kndvi_ds": "kndvi",
          "lst_z": "lst_c", "kndvi_z": "kndvi"}

# Lookup tables kept so predictions can be moved between target spaces exactly.
CLIM = {}
for prim, strict in zip(TARGETS, STRICT_TARGETS):
    raw = RAW_OF[prim]

    # (a) PRIMARY: remove the REGIONAL month climatology, keep county structure,
    #     scale by a SINGLE global sd. A global divisor cannot explode the way a
    #     per-county-month divisor did.
    mmu = panel.loc[train_mask].groupby("month")[raw].mean()
    ds_unscaled = panel[raw] - panel.month.map(mmu)
    gsd = float(ds_unscaled[train_mask].std())
    panel[prim] = (ds_unscaled / gsd).clip(-CLIP_SIGMA, CLIP_SIGMA)

    # (b) STRICT: county x month demeaned, with the sd floored to stop the
    #     near-zero-variance counties from producing absurd values.
    cm = (panel.loc[train_mask].groupby(["adcode", "month"])[raw]
          .agg(["mean", "std"]).reset_index()
          .rename(columns={"mean": "cmu", "std": "csd"}))
    floor = MIN_SD_FRAC * gsd
    n_floored = int((cm.csd.fillna(0) < floor).sum())
    cm["csd_eff"] = cm.csd.fillna(floor).clip(lower=floor)
    panel = panel.merge(cm[["adcode", "month", "cmu", "csd_eff"]],
                        on=["adcode", "month"], how="left")
    z_un = (panel[raw] - panel.cmu) / panel.csd_eff
    n_clip = int((z_un.abs() > CLIP_SIGMA).sum())
    panel[strict] = z_un.clip(-CLIP_SIGMA, CLIP_SIGMA)

    CLIM[prim] = dict(raw=raw, gsd=gsd, mmu=mmu,
                      cmu=panel.groupby(["adcode", "month"]).cmu.first(),
                      csd=panel.groupby(["adcode", "month"]).csd_eff.first())
    panel = panel.drop(columns=["cmu", "csd_eff"])
    print(f"{prim:10} global sd={gsd:.4f} | county sds floored: {n_floored} "
          f"| strict values clipped: {n_clip}")

print("\\n--- PRIMARY targets (deseasonalised, global scale) ---")
print(panel[TARGETS].describe().round(3).to_string())
print("\\n--- STRICT targets (county-demeaned, floored + clipped) ---")
print(panel[STRICT_TARGETS].describe().round(3).to_string())

# ---- variance decomposition: how much signal is spatial vs temporal? ----
print("\\n--- variance decomposition (train period) ---")
print("  tells us how much of any R2 is between-county (easy, spatial) vs")
print("  within-county (hard, temporal).")
for col in TARGETS + STRICT_TARGETS:
    d = panel.loc[train_mask, ["adcode", col]].dropna()
    if not len(d):
        continue
    tot = d[col].var()
    between = d.groupby("adcode")[col].mean().var()
    within = tot - between
    print(f"  {col:10} total={tot:6.3f} | between-county={between/tot*100:5.1f}% "
          f"| within-county={within/tot*100:5.1f}%")

# ---- trend feature (warming + greening; test period sits outside train) ----
panel["year_frac"] = (panel.year - TRAIN_YEARS[0]) / 20.0

'''
s = replace_block(s, old_start, old_end, new_targets, "target construction")

# ─────────────────────────────────────────────── 3. add year_frac to features
s = s.replace(
    'DYN_FEATS = DYN_FEATS + ["moy_sin", "moy_cos"]',
    'DYN_FEATS = DYN_FEATS + ["moy_sin", "moy_cos", "year_frac"]')
print("  patched: year_frac feature")

# ─────────────────────────────────────────── 4. multi-space evaluate()
old_start = '''def evaluate(pred, obs, ci, ti, tag, A_for_moran):'''
old_end = '''print("evaluate() defined")'''

new_eval = '''# Affine maps between target spaces, so a model trained on the PRIMARY target
# can be scored in the STRICT space without retraining.
MONTH_OF_TI = month_of

def build_space_maps():
    maps = {}
    for prim, strict in zip(TARGETS, STRICT_TARGETS):
        c = CLIM[prim]
        mmu_arr = np.array([c["mmu"].get(m, np.nan) for m in range(1, 13)])
        cmu = np.full((N, 13), np.nan, dtype=np.float64)
        csd = np.full((N, 13), np.nan, dtype=np.float64)
        for (a, m), v in c["cmu"].items():
            if a in cidx:
                cmu[cidx[a], m] = v
        for (a, m), v in c["csd"].items():
            if a in cidx:
                csd[cidx[a], m] = v
        maps[prim] = dict(gsd=c["gsd"], mmu=mmu_arr, cmu=cmu, csd=csd,
                          strict=strict)
    return maps

SPACE = build_space_maps()

def to_raw(vals, prim, ci, ti):
    m = SPACE[prim]
    mon = MONTH_OF_TI[ti]
    return vals * m["gsd"] + m["mmu"][mon - 1]

def to_strict(vals, prim, ci, ti):
    m = SPACE[prim]
    mon = MONTH_OF_TI[ti]
    raw = vals * m["gsd"] + m["mmu"][mon - 1]
    return (raw - m["cmu"][ci, mon]) / m["csd"][ci, mon]


def evaluate(pred, obs, ci, ti, tag, A_for_moran):
    """Metric bundle in PRIMARY, RAW and STRICT spaces."""
    out = {}
    out.update({f"all_{k}": v for k, v in
                core_metrics(obs.ravel(), pred.ravel()).items()})
    for j, tname in enumerate(TARGETS):
        for k, v in core_metrics(obs[:, j], pred[:, j]).items():
            out[f"{tname}_{k}"] = v

    # ---- raw + strict spaces (exact affine back-transform) --------------
    for j, prim in enumerate(TARGETS):
        rp = to_raw(pred[:, j], prim, ci, ti)
        ro = to_raw(obs[:, j], prim, ci, ti)
        for k, v in core_metrics(ro, rp).items():
            out[f"raw_{SPACE[prim]['raw']}_{k}"] = v
        sp = to_strict(pred[:, j], prim, ci, ti)
        so = to_strict(obs[:, j], prim, ci, ti)
        for k, v in core_metrics(so, sp).items():
            out[f"strict_{SPACE[prim]['strict']}_{k}"] = v

    df = pd.DataFrame({"ci": ci, "ti": ti})
    for j, tname in enumerate(TARGETS):
        df[f"obs_{tname}"] = obs[:, j]
        df[f"pred_{tname}"] = pred[:, j]
        df[f"res_{tname}"] = pred[:, j] - obs[:, j]
    df["adcode"] = [counties[c] for c in df.ci]
    df["year"] = year_of[df.ti.values]
    df["month"] = month_of[df.ti.values]

    # residual Moran's I on the primary target, averaged over test years
    prim = TARGETS[0]
    Is = []
    for y, g in df.groupby("year"):
        vec = np.full(N, np.nan)
        gm = g.groupby("ci")[f"res_{prim}"].mean()
        vec[gm.index.values] = gm.values
        Is.append(morans_I(vec, A_for_moran))
    out["residual_MoranI"] = float(np.nanmean(Is)) if Is else np.nan

    # within-county R2: removes the between-county component, so it shows
    # whether the model has learned any TEMPORAL signal at all
    dd = df.dropna(subset=[f"obs_{prim}", f"pred_{prim}"]).copy()
    dd["o_d"] = dd[f"obs_{prim}"] - dd.groupby("ci")[f"obs_{prim}"].transform("mean")
    dd["p_d"] = dd[f"pred_{prim}"] - dd.groupby("ci")[f"pred_{prim}"].transform("mean")
    out["within_county_R2"] = core_metrics(dd.o_d, dd.p_d)["R2"]
    out["within_county_PearsonR"] = core_metrics(dd.o_d, dd.p_d)["PearsonR"]

    # shock months (N1), keyed on (ci, ti)
    shock = panel[["ci", "ti", "heat_z"]].copy()
    shock["is_shock"] = shock.heat_z.abs() >= SHOCK_THRESHOLD
    df = df.merge(shock[["ci", "ti", "is_shock"]], on=["ci", "ti"], how="left")
    df["is_shock"] = df.is_shock.fillna(False).astype(bool)
    for lbl, sub in (("shock", df[df.is_shock]), ("calm", df[~df.is_shock])):
        mm = core_metrics(sub[f"obs_{prim}"], sub[f"pred_{prim}"])
        out[f"{lbl}_RMSE"] = mm["RMSE"]
        out[f"{lbl}_R2"] = mm["R2"]
        out[f"{lbl}_n"] = int(len(sub))

    reach = panel.groupby("adcode").reach.first()
    df["reach"] = df.adcode.map(reach)
    for r, sub in df.groupby("reach"):
        out[f"reach_{r}_RMSE"] = core_metrics(
            sub[f"obs_{prim}"], sub[f"pred_{prim}"])["RMSE"]

    out["n_samples"] = int(len(df))
    return out, df
'''
s = replace_block(s, old_start, old_end, new_eval, "multi-space evaluate()")

# ─────────────────────────────────── 5. trivial baselines section
anchor = 'md(r"""\n## 8 · Baseline 1 — DRSEI (Autoencoder + LSTM)'
trivial = '''md(r"""
## 8 · Trivial reference baselines — **the floor**

The previous run had no floor, so there was no way to tell whether a negative
R² meant a broken model or an unpredictable target. These four cost nothing and
settle it:

| Reference | Prediction |
|---|---|
| **Climatology** | 0 — the regional month normal |
| **County climatology** | the county's own train-period mean offset |
| **Persistence** | last month's value |
| **Seasonal naive** | the same month one year earlier |

**Any deep model that fails to beat Persistence is not learning anything.**
These rows are written into `baseline_comparison.csv` alongside the deep models
and tagged `type = trivial`.
""")

code(r\'\'\'
def trivial_predictions(kind, ci, ti):
    """Predictions for the trivial references, in PRIMARY target space."""
    P = np.zeros((len(ci), len(TARGETS)), dtype=np.float64)
    for j, tname in enumerate(TARGETS):
        col = Y[:, :, j]
        if kind == "Climatology":
            P[:, j] = 0.0                              # regional month normal
        elif kind == "CountyClimatology":
            tr_mean = np.nanmean(col[:, tr_t], axis=1)  # train-period offset
            P[:, j] = np.nan_to_num(tr_mean[ci])
        elif kind == "Persistence":
            P[:, j] = np.nan_to_num(col[ci, ti - 1])
        elif kind == "SeasonalNaive":
            P[:, j] = np.nan_to_num(col[ci, np.maximum(ti - 12, 0)])
    return P

TRIVIAL = {}
for kind in ["Climatology", "CountyClimatology", "Persistence", "SeasonalNaive"]:
    pred = trivial_predictions(kind, te_ci, te_ti)
    obs = Y[te_ci, te_ti]
    mt, dfx = evaluate(pred, obs, te_ci, te_ti, "test", A_sp)
    mt.update(model=kind, type="trivial", n_parameters=0, epochs_run=0,
              train_seconds=0.0, best_val_rmse=np.nan)
    TRIVIAL[kind] = mt
    fold = OUTPUT_DIR / "trivial_baselines"
    fold.mkdir(parents=True, exist_ok=True)
    with open(fold / f"{kind}_metrics.json", "w") as f:
        json.dump(mt, f, indent=2, default=float)
    print(f"{kind:20} RMSE={mt['all_RMSE']:.4f}  R2={mt['all_R2']:+.4f}  "
          f"r={mt['all_PearsonR']:+.4f}  withinR2={mt['within_county_R2']:+.4f}")

print("\\n>>> THE BAR: any deep model must beat Persistence on RMSE.")
print(f"    Persistence RMSE = {TRIVIAL['Persistence']['all_RMSE']:.4f}")
best_triv = min(TRIVIAL.values(), key=lambda m: m["all_RMSE"])
print(f"    Best trivial     = {best_triv['model']} at "
      f"{best_triv['all_RMSE']:.4f}")
\'\'\')

'''
s = s.replace(anchor, trivial + anchor)
print("  patched: trivial baselines section")

# ─────────────────────────────────── 6. tag deep models + merge trivial
s = s.replace(
    '''    metrics.update(model=name, n_parameters=int(nparam),
                   epochs_run=len(hist),''',
    '''    metrics.update(model=name, type="deep", n_parameters=int(nparam),
                   epochs_run=len(hist),''')

s = s.replace(
    '''    print(f"\\n--- {name} TEST METRICS ---")
    for k in ["all_RMSE", "all_MAE", "all_R2", "all_PearsonR", "all_WillmottD",
              "all_KGE", "all_Bias", "residual_MoranI", "shock_RMSE",
              "calm_RMSE"]:''',
    '''    print(f"\\n--- {name} TEST METRICS ---")
    for k in ["all_RMSE", "all_MAE", "all_R2", "all_PearsonR", "all_WillmottD",
              "all_KGE", "all_Bias", "within_county_R2", "within_county_PearsonR",
              "residual_MoranI", "shock_RMSE", "calm_RMSE",
              "strict_lst_z_R2", "raw_lst_c_R2"]:''')
print("  patched: deep-model tagging + extended metric printout")

# ─────────────────────────────────── 7. comparison includes trivial
s = s.replace(
    '''comp = pd.DataFrame(ALL).T.reset_index(drop=True)
front = ["model", "n_parameters", "epochs_run", "train_seconds",
         "best_val_rmse", "all_RMSE", "all_MAE", "all_R2", "all_PearsonR",
         "all_WillmottD", "all_KGE", "all_Bias", "residual_MoranI",
         "shock_RMSE", "calm_RMSE"]''',
    '''# trivial references first, then the deep models
rows = list(TRIVIAL.values()) + list(ALL.values())
comp = pd.DataFrame(rows).reset_index(drop=True)
front = ["model", "type", "n_parameters", "epochs_run", "train_seconds",
         "best_val_rmse", "all_RMSE", "all_MAE", "all_R2", "all_PearsonR",
         "all_WillmottD", "all_KGE", "all_Bias", "within_county_R2",
         "within_county_PearsonR", "residual_MoranI", "shock_RMSE",
         "calm_RMSE", "strict_lst_z_R2", "raw_lst_c_R2"]''')

s = s.replace(
    '''models = comp.model.tolist()
palette = ["#4C72B0", "#DD8452", "#55A868"]''',
    '''models = comp.model.tolist()
deep_models = comp.loc[comp.type == "deep", "model"].tolist()
base_pal = ["#BBBBBB", "#999999", "#E15759", "#777777"]     # trivial
deep_pal = ["#4C72B0", "#DD8452", "#55A868"]                 # deep
palette = [base_pal[i % 4] if t == "trivial" else
           deep_pal[max(0, sum(1 for x in comp.type[:i] if x == "deep")) % 3]
           for i, t in enumerate(comp.type)]''')

s = s.replace(
    '''               ("residual_MoranI", "Residual Moran's I (closer to 0 better)"),
               ("shock_RMSE", "Shock-month RMSE (lower better)")]''',
    '''               ("within_county_R2", "Within-county R$^2$ (temporal skill)"),
               ("residual_MoranI", "Residual Moran's I (closer to 0 better)"),
               ("shock_RMSE", "Shock-month RMSE (lower better)")]''')

# persistence reference line on the comparison bars
s = s.replace(
    '''    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel(lab); ax.set_title(f"Baseline comparison — {lab}")
    plt.xticks(rotation=15)''',
    '''    ax.axhline(0, color="k", lw=1)
    if key in ("all_RMSE", "all_MAE", "shock_RMSE"):
        pv = float(comp.loc[comp.model == "Persistence", key].iloc[0])
        ax.axhline(pv, color="crimson", ls="--", lw=2.5,
                   label="Persistence floor")
        ax.legend()
    ax.set_ylabel(lab); ax.set_title(f"Baseline comparison — {lab}")
    plt.xticks(rotation=30, ha="right")''')
print("  patched: comparison table + plots include trivial references")

# ─────────────────────────────────── 8. final summary verdict
s = s.replace(
    '''best = comp.loc[comp.all_RMSE.astype(float).idxmin(), "model"]
print(f"\\nStrongest baseline by test RMSE: {best}")
print("This is the bar the proposed model (PERSIST) must clear.\\n")''',
    '''pers = float(comp.loc[comp.model == "Persistence", "all_RMSE"].iloc[0])
deep = comp[comp.type == "deep"]
best_deep = deep.loc[deep.all_RMSE.astype(float).idxmin()]
print(f"\\nPersistence floor       : RMSE {pers:.4f}")
print(f"Best deep baseline      : {best_deep['model']} at "
      f"{float(best_deep.all_RMSE):.4f}")
if float(best_deep.all_RMSE) < pers:
    print(f"  -> deep models BEAT persistence by "
          f"{(pers - float(best_deep.all_RMSE))/pers*100:.1f}%. Task has signal.")
else:
    print("  -> deep models DO NOT beat persistence. The target still lacks")
    print("     learnable signal; escalate to seasonal aggregation before")
    print("     touching architectures.")
print("This is the bar the proposed model (PERSIST) must clear.\\n")''')
print("  patched: verdict vs persistence floor")

# ─────────────────────────────────── 9. title note
s = s.replace(
    "> **Only the CONFIG cell below needs editing.**",
    """### Revision note

The first run returned negative R² and near-zero correlation for all three
models. Diagnosis: the fault was the **target definition**, not the code.

| Problem | Fix |
|---|---|
| Per-county-month standardisation divided by near-zero sds, producing a **61-sigma** kNDVI value that dominated every squared-error metric | Single **global** scale for the primary target; county sds floored and values clipped for the strict target |
| County-demeaning removed nearly all predictable signal (measured monthly lag-1 autocorrelation is only **0.155**), capping achievable R² at ≈0.03 | Primary target is now **deseasonalised state** — regional seasonal cycle removed, county structure retained |
| No floor, so bad metrics were uninterpretable | **Four trivial baselines** added: climatology, county-climatology, persistence, seasonal-naive |
| Warming/greening trend shifted the test distribution (test variance ≈2× train) | `year_frac` trend feature added |
| No way to see whether skill was spatial or temporal | **Variance decomposition** + **within-county R²** reported |

> **Only the CONFIG cell below needs editing.**""")
print("  patched: revision note in title")

SRC.write_text(s)
print(f"\nwrote {SRC} ({len(s)/1024:.0f} KB)")
