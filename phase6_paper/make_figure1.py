#!/usr/bin/env python3
"""Figure 1 - overall workflow of the proposed PERSIST framework."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = ("/projects/sandbox/Assesment_of_Ecological_resilience_Yangtze/"
       "phase6_paper/figures/Figure1_PERSIST_workflow.png")

# palette: (header fill, body fill, edge)
C = {
    "navy":   ("#1F3B63", "#FFFFFF", "#1F3B63"),
    "green":  ("#1E6B4F", "#F2FAF6", "#1E6B4F"),
    "purple": ("#6B3FA0", "#F7F2FC", "#6B3FA0"),
    "gold":   ("#8A6D1F", "#FDF8E8", "#8A6D1F"),
    "rose":   ("#B03A48", "#FDF0F1", "#B03A48"),
    "blue":   ("#2166A5", "#F0F6FC", "#2166A5"),
    "teal":   ("#17696B", "#EEF7F7", "#17696B"),
}

fig, ax = plt.subplots(figsize=(22, 13.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 66)
ax.axis("off")

HDR_FS, ITEM_FS, HDR_H = 13.5, 11.6, 6.6


def block(x, y, w, h, num, title, items, key):
    """title may contain a newline; the header band is a fixed height so all
    blocks line up whether the title wraps to one line or two."""
    hc, bc, ec = C[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
                                linewidth=2.0, edgecolor=ec, facecolor=bc, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - HDR_H), w, HDR_H,
                                boxstyle="round,pad=0,rounding_size=0.7",
                                linewidth=0, facecolor=hc, zorder=3))
    ax.add_patch(plt.Rectangle((x, y + h - HDR_H), w, 1.0, facecolor=hc,
                               edgecolor="none", zorder=3))
    ax.text(x + w / 2, y + h - HDR_H / 2, f"{num}. {title}", ha="center",
            va="center", fontsize=HDR_FS, fontweight="bold", color="white",
            zorder=4, linespacing=1.3)
    n = len(items)
    top, bot = y + h - HDR_H - 1.8, y + 1.5
    step = (top - bot) / max(n - 1, 1) if n > 1 else 0
    for i, it in enumerate(items):
        yy = top - i * step if n > 1 else (top + bot) / 2
        ax.text(x + w / 2, yy, it, ha="center", va="center", fontsize=ITEM_FS,
                color="#1A1A1A", zorder=4, linespacing=1.25)


def arrow(x1, y1, x2, y2, dashed=False, rad=0.0, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=26, linewidth=2.2, color=color,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", zorder=5,
                                 shrinkA=0, shrinkB=0))


# ─────────────────────────────── top row ────────────────────────────────
TY, TH, W = 38.5, 26.0, 15.0
xs = [1.0, 17.0, 33.0, 49.0, 65.0, 81.0]

block(xs[0], TY, W, TH, 1, "Data\nAcquisition", [
    "MODIS LST / NDVI\n(MOD11A2, MOD13Q1)",
    "TerraClimate forcing\n(21 variables)",
    "Copernicus DEM 30 m\nterrain derivatives",
    "WOKAM karst extent",
    "Harmonised nighttime\nlights",
    "DataV county boundaries\n(GB/T 2260)"], "navy")

block(xs[1], TY, W, TH, 2, "Panel &\nTarget", [
    "1,068 counties x 252\nmonths (2000-2020)",
    "Deseasonalise vs\ntrain-only climatology",
    "Single global scale\n(LST 3.605 °C)",
    "Forward 3-month\nmean anomaly",
    "Embargoed splits\n15 / 3 / 3 years",
    "166 / 34 / 34 windows"], "green")

block(xs[2], TY, W, TH, 3, "Process-Informed\nState", [
    "Monthly GRU\n(12-month lookback)",
    "Annual GRU on\nyear-over-year deltas",
    "Static context encoder\n(terrain, karst, area)",
    "Forcing / state split\n32 dynamic features",
    "Scaled residual fusion",
    r"$\rightarrow$  latent  $h_{i,t}$"], "purple")

block(xs[3], TY, W, TH, 4, "Dual-Graph\nMessage Passing", [
    "Contiguity graph\n3,032 undirected edges",
    "Hydrological graph\n3,032 directed edges",
    "Masked graph attention\n(2 heads)",
    "Gated residual fusion\n(scale init 0.30)",
    "Node-chunked batching\n3 x 356 counties",
    "Per-chunk Laplacians"], "teal")

block(xs[4], TY, W, TH, 5, "Lithology-Adaptive\nMoE", [
    "4 experts",
    "Soft gate on static\nterrain / karst context",
    "karst fraction, relief,\nslope, elevation",
    "Load-balancing\nentropy penalty",
    "Expert-mixed\ncoefficient heads",
    "Handles karst and\nnon-karst jointly"], "gold")

block(xs[5], TY, W, TH, 6, "Response\nDecoder", [
    r"$\rho$  persistence  (0,1)",
    r"$\sigma$  seasonal carry  (0,1)",
    r"$\kappa$  forcing sensitivity",
    r"$\delta$  offset",
    r"$+\ a_{res}\cdot$ direct$(z)$",
    "Nests SeasonalNaive and\nTrailingPersistence"], "rose")

# ────────────────────────────── bottom row ──────────────────────────────
BY, BH = 6.0, 26.0

block(xs[5], BY, W, BH, 7, "Ecological\nLoss", [
    "Masked prediction MSE",
    "Feature reconstruction",
    r"Temporal smoothness of $\rho$",
    "Graph Laplacian on\nRESIDUALS (every step)",
    "Asymmetric runaway\npenalty",
    "Expert load balance"], "gold")

block(xs[4], BY, W, BH, 8, "Training\nProtocol", [
    "AdamW, cosine schedule\nwith warmup",
    "EMA weights\n(decay 0.998)",
    "Gradient clipping",
    "Early stopping on\nvalidation RMSE",
    "3 seeds -> ensemble",
    "Seed-noise band\nreported"], "blue")

block(xs[3], BY, W, BH, 9, "Evaluation", [
    "4 trivial references",
    "3 deep baselines\n(DRSEI, STGCN, TFT)",
    "7 component ablations",
    "Pooled vs within-county\nR-squared",
    "Stride-3 non-overlapping\ncheck",
    "Residual Moran's I,\nshock-window RMSE"], "blue")

block(xs[2], BY, W, BH, 10, "Resilience\nAssessment", [
    r"Recovery rate  $1-\rho$",
    r"Resistance  $1-|\kappa|$",
    r"Adaptability: drift in $\rho$",
    "County choropleth maps",
    "Reach and karst\ngradients",
    "n1_coefficients.csv"], "green")

# ──────────────────────── study-area side panel ─────────────────────────
sx, sy, sw, sh = 1.0, 9.0, 28.0, 20.0
ax.add_patch(FancyBboxPatch((sx, sy), sw, sh,
                            boxstyle="round,pad=0,rounding_size=0.7",
                            linewidth=2.4, edgecolor="#17696B",
                            facecolor="#DCEEEE", zorder=2))
ax.text(sx + sw / 2, sy + sh - 2.8, "Yangtze River Economic Belt",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color="#0E4C4E")
ax.text(sx + sw / 2, sy + sh - 6.8,
        "1,068 county-level units  ·  2,052,264 km$^2$\n"
        "269,136 county-months  ·  2000-2020",
        ha="center", va="center", fontsize=12.2, color="#123", linespacing=1.5)
for i, (lab, n, km) in enumerate([("Upstream", 438, "1,128,276"),
                                  ("Midstream", 325, "564,904"),
                                  ("Downstream", 305, "359,084")]):
    ax.text(sx + sw / 2, sy + 8.0 - i * 2.9,
            f"{lab}:  {n} counties  ·  {km} km$^2$",
            ha="center", va="center", fontsize=11.8, color="#123")

# ───────────────────────────────  arrows  ───────────────────────────────
for i in range(5):                                    # 1 -> 6 across the top
    arrow(xs[i] + W, TY + TH / 2, xs[i + 1], TY + TH / 2)
arrow(xs[5] + W / 2, TY, xs[5] + W / 2, BY + BH)      # 6 -> 7 down
for i in [5, 4, 3]:                                   # 7 -> 10 leftwards
    arrow(xs[i], BY + BH / 2, xs[i - 1] + W, BY + BH / 2)
arrow(xs[0] + W / 2, sy + sh, xs[0] + W / 2, TY)      # study area -> data
arrow(xs[2], BY + BH / 2, sx + sw, BY + BH / 2,       # 10 -> study area
      dashed=True, color="#B03A48")
ax.text(24.0, 5.2, "resilience maps returned to the study area",
        fontsize=11.5, color="#B03A48", style="italic", ha="center", va="center")

fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", OUT)
