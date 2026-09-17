#!/usr/bin/env python3
"""Smoke test PERSIST_Phase5_v3.ipynb.

Patching is regex-based and VERIFIED: if any patch matches nothing we abort.
(The v2 harness used exact-string keys that silently no-oped against the
notebook's column-aligned assignments and launched the full run by mistake.)

Stages:
    core       trivial + 3 deep baselines + PERSIST, H=3, 2 seeds
    ablations  + all 7 ablations, H=3, 1 seed
    h1         HORIZON=1 backwards-compatibility check (reproduces v2's task)
"""
import re
import sys
import time
import traceback

import nbformat

NB = "PERSIST_Phase5_v3.ipynb"
LOCAL = "/projects/sandbox/Assesment_of_Ecological_resilience_Yangtze"

STAGE = sys.argv[1] if len(sys.argv) > 1 else "core"
CFG = {
    "core":      dict(out="smoke_v3_core", base="True",  abl="False",
                      ns=2, ep=2, nc=60, hz=3),
    "ablations": dict(out="smoke_v3_abl",  base="False", abl="True",
                      ns=1, ep=2, nc=40, hz=3),
    "h1":        dict(out="smoke_v3_h1",   base="True",  abl="False",
                      ns=1, ep=1, nc=40, hz=1),
}[STAGE]
OUT = f"/projects/sandbox/yreb_resilience/{CFG['out']}"

PATCH = [
    (r"^MOUNT_DRIVE\s*=.*$",   "MOUNT_DRIVE = False"),
    (r"^REPO_ROOT\s*=.*$",     f'REPO_ROOT   = Path("{LOCAL}")'),
    (r"^OUTPUT_DIR\s*=.*$",    f'OUTPUT_DIR    = Path("{OUT}")'),
    (r"^SMOKE_TEST\s*=.*$",    "SMOKE_TEST     = True"),
    (r"^RUN_BASELINES\s*=.*$", f"RUN_BASELINES  = {CFG['base']}"),
    (r"^RUN_ABLATIONS\s*=.*$", f"RUN_ABLATIONS  = {CFG['abl']}"),
    (r"^HORIZON\s*=.*$",       f"HORIZON    = {CFG['hz']}"),
    (r"^DPI\s*=.*$",           "DPI       = 80"),
    (r"^NODE_CHUNKS\s*=.*$",   "NODE_CHUNKS   = 2"),
    (r"^EPOCHS\s*=.*$",        f"EPOCHS        = {CFG['ep']}"),
    (r"^EPOCHS_BASE\s*=.*$",   f"EPOCHS_BASE   = {CFG['ep']}"),
    (r"^PATIENCE\s*=.*$",      "PATIENCE      = 99"),
    (r"^PATIENCE_BASE\s*=.*$", "PATIENCE_BASE = 99"),
    (r"^WARMUP_EPOCHS\s*=.*$", "WARMUP_EPOCHS = 1"),
    (r"^(\s*)keep = sorted\(panel\.adcode\.unique\(\)\)\[:\d+\]$",
     rf"\1keep = sorted(panel.adcode.unique())[:{CFG['nc']}]"),
    (r"^EP  = 2 if SMOKE_TEST else EPOCHS$",       "EP  = EPOCHS"),
    (r"^EPB = 2 if SMOKE_TEST else EPOCHS_BASE$",  "EPB = EPOCHS_BASE"),
    (r"^NS  = 1 if SMOKE_TEST else N_SEEDS$",      f"NS  = {CFG['ns']}"),
]

nb = nbformat.read(NB, as_version=4)
cells = [c.source for c in nb.cells if c.cell_type == "code"]
print(f"stage={STAGE}  cells={len(cells)}  H={CFG['hz']} counties={CFG['nc']} "
      f"epochs={CFG['ep']} seeds={CFG['ns']} baselines={CFG['base']} "
      f"ablations={CFG['abl']}\n")

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
print(f"all {len(PATCH)} patches applied\n")

g = {"__name__": "__main__"}
g["display"] = lambda *a, **k: [
    print(x.to_string() if hasattr(x, "to_string") else x) for x in a]

t_start = time.time()
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

print("\n" + "=" * 66)
print(f"PERSIST v3 SMOKE PASSED  stage={STAGE}  ({time.time()-t_start:.0f}s)")
print("=" * 66)
