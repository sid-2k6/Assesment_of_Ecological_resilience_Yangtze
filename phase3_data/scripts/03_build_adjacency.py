#!/usr/bin/env python3
"""Build the county adjacency graph for the YREB — the relational structure the
Phase-2 GNN novelty requires, and the gap identified in the survey (spatial
autocorrelation is confirmed in the literature but never modelled).

Produces queen contiguity (shared edge OR shared vertex) plus a k-nearest
centroid fallback so no county is left isolated (islands, e.g. Chongming).

Outputs:
  data/interim/adjacency_edges.csv    undirected edge list
  data/interim/adjacency_stats.json   graph diagnostics
"""
import json
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd

OUT = pathlib.Path("data/interim")
CTY = OUT / "yreb_counties_datav.gpkg"
K_FALLBACK = 3


def main():
    gdf = gpd.read_file(CTY).reset_index(drop=True)
    gdf["idx"] = gdf.index
    n = len(gdf)
    print(f"counties: {n}")

    # --- queen contiguity via spatial join on 'touches' -----------------
    # Buffer by a tiny epsilon to absorb sliver gaps in the source geometry;
    # without this, digitisation noise silently drops real neighbours.
    eps = 50  # metres (CRS is Albers, units=m)
    probe = gdf[["idx", "geometry"]].copy()
    probe["geometry"] = probe.geometry.buffer(eps)

    j = gpd.sjoin(probe, gdf[["idx", "geometry"]], predicate="intersects",
                  how="inner", lsuffix="l", rsuffix="r")
    edges = j.loc[j.idx_l != j.idx_r, ["idx_l", "idx_r"]].copy()

    # undirected, deduplicated
    a = np.minimum(edges.idx_l.values, edges.idx_r.values)
    b = np.maximum(edges.idx_l.values, edges.idx_r.values)
    und = pd.DataFrame({"u": a, "v": b}).drop_duplicates()
    print(f"queen edges (eps={eps} m): {len(und)}")

    deg = pd.concat([und.u, und.v]).value_counts().reindex(range(n), fill_value=0)
    isolated = deg[deg == 0].index.tolist()
    print(f"isolated counties before fallback: {len(isolated)}")

    # --- k-nearest centroid fallback for isolates ----------------------
    if isolated:
        cent = gdf.geometry.centroid
        xy = np.c_[cent.x.values, cent.y.values]
        add = []
        for i in isolated:
            d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
            d[i] = np.inf
            for jn in np.argsort(d)[:K_FALLBACK]:
                add.append((min(i, jn), max(i, jn)))
            print(f"  linked isolate {gdf.at[i,'name_zh']} "
                  f"({gdf.at[i,'adcode']}) to {K_FALLBACK} nearest")
        und = pd.concat([und, pd.DataFrame(add, columns=["u", "v"])]
                        ).drop_duplicates()

    deg = pd.concat([und.u, und.v]).value_counts().reindex(range(n), fill_value=0)

    # --- connectivity check (BFS) --------------------------------------
    adj = {i: set() for i in range(n)}
    for u, v in und.itertuples(index=False):
        adj[u].add(v)
        adj[v].add(u)
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    comps.sort(key=len, reverse=True)

    # --- attach codes + cross-reach flags ------------------------------
    m = gdf.set_index("idx")
    out = und.copy()
    out["u_adcode"] = m.loc[out.u, "adcode"].values
    out["v_adcode"] = m.loc[out.v, "adcode"].values
    out["u_reach"] = m.loc[out.u, "reach"].values
    out["v_reach"] = m.loc[out.v, "reach"].values
    out["u_prov"] = m.loc[out.u, "province"].values
    out["v_prov"] = m.loc[out.v, "province"].values
    out["cross_reach"] = out.u_reach != out.v_reach
    out["cross_province"] = out.u_prov != out.v_prov
    out.to_csv(OUT / "adjacency_edges.csv", index=False)

    stats = {
        "n_counties": int(n),
        "n_edges": int(len(out)),
        "mean_degree": float(deg.mean()),
        "median_degree": float(deg.median()),
        "min_degree": int(deg.min()),
        "max_degree": int(deg.max()),
        "density": float(2 * len(out) / (n * (n - 1))),
        "n_components": len(comps),
        "largest_component": len(comps[0]),
        "cross_reach_edges": int(out.cross_reach.sum()),
        "cross_province_edges": int(out.cross_province.sum()),
        "epsilon_m": eps,
        "k_fallback": K_FALLBACK,
    }
    (OUT / "adjacency_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n{'='*58}")
    for k, v in stats.items():
        print(f"  {k:22} {v}")
    if stats["n_components"] > 1:
        print(f"\n  WARNING: graph not fully connected "
              f"({stats['n_components']} components); "
              f"sizes {[len(c) for c in comps[:6]]}")
    else:
        print("\n  graph is fully connected -> safe for message passing")
    print(f"\nwrote -> {OUT}/adjacency_edges.csv")


if __name__ == "__main__":
    main()
