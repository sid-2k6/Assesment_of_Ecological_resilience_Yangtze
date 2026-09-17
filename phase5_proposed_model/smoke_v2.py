#!/usr/bin/env python3
"""Smoke test PERSIST_Phase5_v2.ipynb.

Patches the CONFIG cell down to a tiny local budget and executes every code
cell in order.

IMPORTANT: patching is regex-based and *verified*. If any patch fails to
match, we abort immediately. (v1 of this harness used exact-string keys that
silently no-oped against the notebook's column-aligned assignments, so the
"smoke test" actually launched the full 1,068-county / 150-epoch / 3-seed /
8-config run and hit the command timeout.)

Usage:
    python smoke_v2.py            # stage A: PERSIST only, 2 seeds
    python smoke_v2.py ablations  # stage B: PERSIST + all 7 ablations, 1 seed
"""
import re
import sys
import time
import traceback

import nbformat

NB = "PERSIST_Phase5_v2.ipynb"
LOCAL = "/projects/sandbox/Assesment_of_Ecological_resilience_Yangtze"

STAGE = "ablations" if len(sys.argv) > 1 and sys.argv[1] == "ablations" else "core"
if STAGE == "ablations":
    OUT, ABL, NS, EPOCHS, NCOUNTY = "smoke_outputs_v2_abl", "True", 1, 2, 40
else:
    OUT, ABL, NS, EPOCHS, NCOUNTY = "smoke_outputs_v2", "False", 2, 3, 60
OUT = f"/projects/sandbox/yreb_resilience/{OUT}"
BASE = "/projects/sandbox/yreb_resilience/smoke_outputs/baseline_comparison.csv"

# (regex, replacement) -- applied per code cell with re.MULTILINE
PATCH = [
    (r"^MOUNT_DRIVE\s*=.*$",   "MOUNT_DRIVE = False"),
    (r'^REPO_ROOT\s*=.*$',     f'REPO_ROOT   = Path("{LOCAL}")'),
    (r"^OUTPUT_DIR\s*=.*$",    f'OUTPUT_DIR    = Path("{OUT}")'),
    (r"^BASELINE_CSV\s*=.*$",  f'BASELINE_CSV  = Path("{BASE}")'),
    (r"^SMOKE_TEST\s*=.*$",    "SMOKE_TEST    = True"),
    (r"^RUN_ABLATIONS\s*=.*$", f"RUN_ABLATIONS = {ABL}"),
    (r"^DPI\s*=.*$",           "DPI       = 80"),
    (r"^NODE_CHUNKS\s*=.*$",   "NODE_CHUNKS   = 2"),
    (r"^EPOCHS\s*=.*$",        f"EPOCHS        = {EPOCHS}"),
    (r"^PATIENCE\s*=.*$",      "PATIENCE      = 99"),
    (r"^WARMUP_EPOCHS\s*=.*$", "WARMUP_EPOCHS = 1"),
    (r"^(\s*)keep = sorted\(panel\.adcode\.unique\(\)\)\[:\d+\]$",
     rf"\1keep = sorted(panel.adcode.unique())[:{NCOUNTY}]"),
    (r"^EP = 2 if SMOKE_TEST else EPOCHS$", "EP = EPOCHS"),
    (r"^NS = 1 if SMOKE_TEST else N_SEEDS$", f"NS = {NS}"),
]

nb = nbformat.read(NB, as_version=4)
cells = [c.source for c in nb.cells if c.cell_type == "code"]
print(f"stage={STAGE}  cells={len(cells)}  counties={NCOUNTY} "
      f"epochs={EPOCHS} seeds={NS} ablations={ABL}\n")

hits = {p: 0 for p, _ in PATCH}
patched = []
for src in cells:
    for pat, rep in PATCH:
        src, n = re.subn(pat, rep, src, flags=re.MULTILINE)
        hits[pat] += n
    patched.append(src)

missed = [p for p, n in hits.items() if n == 0]
if missed:
    print("!!! ABORT - these patches matched nothing:")
    for p in missed:
        print("   ", p)
    sys.exit(2)
print("all %d patches applied: %s\n" % (
    len(PATCH), {p.split("\\")[0][1:14]: n for p, n in hits.items()}))

g = {"__name__": "__main__"}
g["display"] = lambda *a, **k: [
    print(x.to_string() if hasattr(x, "to_string") else x) for x in a]

for i, src in enumerate(patched, 1):
    head = next((l for l in src.splitlines()
                 if l.strip() and not l.strip().startswith("#")), "")[:66]
    print(f"── cell {i:>2}/{len(patched)}  {head}", flush=True)
    t0 = time.time()
    try:
        exec(compile(src, f"<cell {i}>", "exec"), g)
    except Exception:
        print(f"\n!!! FAILED at cell {i}\n")
        traceback.print_exc()
        sys.exit(1)
    dt = time.time() - t0
    if dt > 5:
        print(f"   [cell {i} took {dt:.1f}s]", flush=True)

print("\n" + "=" * 62)
print(f"PERSIST v2 SMOKE TEST PASSED (stage={STAGE})")
print("=" * 62)
