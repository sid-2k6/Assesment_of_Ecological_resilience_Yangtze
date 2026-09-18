#!/usr/bin/env python3
"""Figure 2 - overall architecture of the proposed PERSIST framework."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = ("/projects/sandbox/Assesment_of_Ecological_resilience_Yangtze/"
       "phase6_paper/figures/Figure2_PERSIST_architecture.png")

fig, ax = plt.subplots(figsize=(15.0, 10.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.axis("off")

FS_T, FS_B, FS_S = 11.8, 10.4, 9.4
COL = {
    "in":   ("#1F3B63", "#EAF0F7"),
    "enc":  ("#6B3FA0", "#F4EEFB"),
    "grf":  ("#17696B", "#E8F4F4"),
    "moe":  ("#8A6D1F", "#FBF5E4"),
    "dec":  ("#B03A48", "#FCEFF0"),
    "out":  ("#1E6B4F", "#EAF6F1"),
    "loss": ("#2166A5", "#EDF3FA"),
}


def box(x, y, w, h, title, sub=None, shape=None, key="enc", fs=None):
    """Stack title / sub-lines / shape on an even line grid so multi-line
    subtitles can never collide with the title."""
    ec, fc = COL[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                linewidth=1.8, edgecolor=ec, facecolor=fc, zorder=2))
    lines = [(title, "bold", fs or FS_T, ec, "normal")]
    if sub:
        for s in sub.split("\n"):
            lines.append((s, "normal", FS_B, "#1A1A1A", "normal"))
    if shape:
        lines.append((shape, "normal", FS_S, "#555555", "italic"))
    n = len(lines)
    lh = h / (n + 0.55)
    top = y + h - lh * 0.80
    for i, (txt, wt, size, col, st) in enumerate(lines):
        ax.text(x + w / 2, top - i * lh, txt, ha="center", va="center",
                fontsize=size, fontweight=wt, color=col, style=st, zorder=4)


def arr(x1, y1, x2, y2, rad=0.0, color="#333333", dashed=False, lw=1.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=19, linewidth=lw, color=color,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}",
                                 zorder=5, shrinkA=0, shrinkB=0))


L, M, R = 2.0, 34.0, 66.0        # column x
WL = WM = 29.0
WR = 32.0
cL, cM, cR = L + WL / 2, M + WM / 2, R + WR / 2

# ── row 1: inputs ─────────────────────────────────────────────────────────
box(L, 89, WL, 9.5, "Dynamic input window",
    "32 features x 12 months", "(B, 12, N, 32)", "in")
box(M, 89, WM, 9.5, "Annual input window",
    "64 features x 3 years", "(B, 3, N, 64)", "in")
box(R, 89, WR, 9.5, "Static physiographic input",
    "elevation, relief, slope,\nroughness, karst fraction", "(N, 8)", "in")

# ── row 2: primary encoders ───────────────────────────────────────────────
box(L, 76, WL, 9.5, "Monthly GRU   (N5)",
    "GRU(32 $\\rightarrow$ 96), 2 layers,\ndropout 0.20", "(B, N, 96)", "enc")
box(M, 76, WM, 9.5, "Annual GRU   (N5)",
    "GRU(64 $\\rightarrow$ 96) on\nyear-over-year deltas", "(B, N, 96)", "enc")
box(R, 76, WR, 9.5, "Static context encoder",
    "Linear(8 $\\rightarrow$ 96) + ELU", "(N, 96)", "enc")

# ── row 3: secondary encoders + attribution ───────────────────────────────
box(L, 64, WL, 9.0, "State encoder",
    "Linear(5 $\\rightarrow$ 96) + ELU + Linear", "(B, N, 96)", "enc")
box(M, 64, WM, 9.0, "Forcing encoder",
    "Linear(22 $\\rightarrow$ 96) + ELU + Linear", "(B, N, 96)", "enc")
box(R, 64, WR, 9.0, "Attribution head   (N6)",
    "Linear(192 $\\rightarrow$ 4) + softmax,\ngates the four streams", "(B, N, 4)", "moe")

# ── row 4: fusion ─────────────────────────────────────────────────────────
box(L, 53, 61, 8.0, "Scaled residual fusion",
    r"$h = h_{month} + a_{year}h_{annual} + a_{state}h_{state}$"
    "     ($a$ learnable, init 0.10)", None, "enc")
box(R, 53, WR, 8.0, "Lithology expert gate   (N3)",
    "Linear(8 $\\rightarrow$ 4) + softmax", "(N, 4)", "moe")

# ── row 5: graphs ─────────────────────────────────────────────────────────
box(L, 40, WL, 9.5, "Spatial graph attention  (N2b)",
    "2 heads, county contiguity,\n3,032 undirected edges", "(B, N, 96)", "grf", fs=11.0)
box(M, 40, WM, 9.5, "Hydro graph attention  (N2a)",
    "2 heads, directed flow,\n3,032 directed edges", "(B, N, 96)", "grf", fs=11.0)
box(R, 40, WR, 9.5, "Ecological objective   (N4)",
    "prediction + reconstruction\n+ $\\rho$ smoothness + Laplacian\non residuals + load balance",
    None, "loss", fs=11.0)

# ── row 6: gated graph fusion ─────────────────────────────────────────────
box(L, 30, 61, 7.5, "Gated graph fusion",
    r"$h \leftarrow h + a_{sp}m_{sp} + a_{hy}m_{hy}$"
    "     ($a$ init 0.30, learnable)", None, "grf")

# ── row 7: concatenation (full width) ─────────────────────────────────────
box(L, 20, 96, 7.5, "Concatenation with static context",
    r"$z = [\,h \;;\; c\,]$", "(B, N, 192)", "enc")

# ── row 8: four heads ─────────────────────────────────────────────────────
HW, HX = 22.0, [2.0, 26.0, 50.0, 74.0]
box(HX[0], 8.5, HW, 9.5, "Coefficient experts  (N1)",
    "4 x [Linear(192$\\rightarrow$96)\n+ ELU + Dropout\n+ Linear(96$\\rightarrow$8)]",
    None, "dec", fs=10.8)
box(HX[1], 8.5, HW, 9.5, "Forcing-effect head",
    "Linear(192$\\rightarrow$96) + ELU\n+ Linear(96$\\rightarrow$2)",
    "(B, N, 2)", "dec", fs=10.8)
box(HX[2], 8.5, HW, 9.5, "Free residual head  (R1)",
    "Linear + ELU + Dropout\n" r"scaled by $a_{res}$ (init 0.30)",
    "(B, N, 2)", "dec", fs=10.8)
box(HX[3], 8.5, HW, 9.5, "Reconstruction head  (N4)",
    "Linear(96 $\\rightarrow$ 32)\nfeature reconstruction",
    "(B, N, 32)", "loss", fs=10.8)

# ── row 9: output ─────────────────────────────────────────────────────────
box(L, 0.4, 96, 6.6, "Bounded coefficients and seasonal anomaly prediction",
    r"$\rho,\varsigma \in (0,1)$,  $\kappa \in (-1,1)$,  $\delta$   "
    r"$\Rightarrow$   $\hat{y} = \rho\,y_{prev} + \varsigma\,y_{seas}"
    r" + \kappa\,f_{eff} + \delta + a_{res}\,\mathrm{direct}(z)$",
    None, "out")

# ── arrows ────────────────────────────────────────────────────────────────
for c in (cL, cM, cR):
    arr(c, 89, c, 85.5)                       # inputs -> encoders
    arr(c, 76, c, 73.0)                       # encoders -> row 3
arr(cL, 64, cL, 61.0)                         # state enc -> fusion
arr(cM, 64, cM, 61.0)                         # forcing enc -> fusion
arr(cR, 64, cR, 61.0)                         # N6 -> N3
arr(cL, 53, cL, 49.5)                         # fusion -> spatial graph
arr(cM, 53, cM, 49.5)                         # fusion -> hydro graph
arr(cL, 40, cL, 37.5)                         # graphs -> gated fusion
arr(cM, 40, cM, 37.5)
arr(cL + 14.5, 30, cL + 14.5, 27.5)           # gated fusion -> z
arr(cR, 53, cR, 49.5)                         # N3 -> objective column
arr(cR, 40, cR, 27.5)                         # right column -> z
for hx in HX:
    arr(hx + HW / 2, 20, hx + HW / 2, 18.0)   # z -> heads
arr(HX[0] + HW / 2, 8.5, HX[0] + HW / 2, 7.0)          # experts -> output
arr(HX[1] + HW / 2, 8.5, HX[1] + HW / 2, 7.0)
arr(HX[2] + HW / 2, 8.5, HX[2] + HW / 2, 7.0)
arr(HX[3] + HW / 2, 8.5, HX[3] + HW / 2, 7.0,
    dashed=True, color="#2166A5")                      # recon -> objective

ax.text(50, 99.4, "PERSIST — process-informed spatiotemporal architecture",
        ha="center", va="center", fontsize=14.0, fontweight="bold", color="#222222")

fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", OUT)
