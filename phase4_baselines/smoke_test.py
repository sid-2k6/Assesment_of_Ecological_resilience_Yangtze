#!/usr/bin/env python3
"""Smoke test the generated notebook: extract code cells, patch the CONFIG for
local sandbox paths + tiny subset, execute in order, report the first failure.
"""
import re
import sys
import traceback

import nbformat

NB = "PERSIST_Phase4_Baselines.ipynb"
LOCAL_ROOT = "/projects/sandbox/Assesment_of_Ecological_resilience_Yangtze"
OUT_ROOT = "/projects/sandbox/yreb_resilience/smoke_outputs"

nb = nbformat.read(NB, as_version=4)
cells = [c.source for c in nb.cells if c.cell_type == "code"]
print(f"{len(cells)} code cells\n")

PATCH = {
    'MOUNT_DRIVE = True': 'MOUNT_DRIVE = False',
    'REPO_ROOT   = Path("/content/drive/MyDrive/Assesment_of_Ecological_resilience_Yangtze")':
        f'REPO_ROOT   = Path("{LOCAL_ROOT}")',
    'OUTPUT_DIR    = REPO_ROOT / "outputs"': f'OUTPUT_DIR    = Path("{OUT_ROOT}")',
    'SMOKE_TEST = False': 'SMOKE_TEST = True',
    'DPI       = 300': 'DPI       = 100',      # keep the smoke test fast
}

g = {"__name__": "__main__"}
# display() is IPython-only; provide a no-op equivalent
g["display"] = lambda *a, **k: [print(x.to_string() if hasattr(x, "to_string")
                                      else x) for x in a]

for i, src in enumerate(cells, 1):
    for a, b in PATCH.items():
        src = src.replace(a, b)
    head = next((l for l in src.splitlines()
                 if l.strip() and not l.strip().startswith("#")), "")[:64]
    print(f"── cell {i:>2}/{len(cells)}  {head}")
    try:
        exec(compile(src, f"<cell {i}>", "exec"), g)
    except Exception:
        print(f"\n!!! FAILED at cell {i}\n")
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 60)
print("SMOKE TEST PASSED — all cells executed")
print("=" * 60)
