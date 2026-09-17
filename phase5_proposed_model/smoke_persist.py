#!/usr/bin/env python3
"""Smoke test the PERSIST notebook: patch CONFIG for a tiny local run and
execute every code cell in order."""
import sys
import traceback

import nbformat

NB = "PERSIST_Phase5_Proposed_Model.ipynb"
LOCAL = "/projects/sandbox/Assesment_of_Ecological_resilience_Yangtze"
OUT = "/projects/sandbox/yreb_resilience/smoke_outputs"   # has baseline csv

nb = nbformat.read(NB, as_version=4)
cells = [c.source for c in nb.cells if c.cell_type == "code"]
print(f"{len(cells)} code cells\n")

PATCH = {
    "MOUNT_DRIVE = True": "MOUNT_DRIVE = False",
    'REPO_ROOT   = Path("/content/drive/MyDrive/Assesment_of_Ecological_resilience_Yangtze")':
        f'REPO_ROOT   = Path("{LOCAL}")',
    'OUTPUT_DIR    = REPO_ROOT / "outputs"': f'OUTPUT_DIR    = Path("{OUT}")',
    "SMOKE_TEST = False": "SMOKE_TEST = True",
    "DPI       = 300": "DPI       = 100",
}

g = {"__name__": "__main__"}
g["display"] = lambda *a, **k: [
    print(x.to_string() if hasattr(x, "to_string") else x) for x in a]

for i, src in enumerate(cells, 1):
    for a, b in PATCH.items():
        src = src.replace(a, b)
    head = next((l for l in src.splitlines()
                 if l.strip() and not l.strip().startswith("#")), "")[:66]
    print(f"── cell {i:>2}/{len(cells)}  {head}")
    try:
        exec(compile(src, f"<cell {i}>", "exec"), g)
    except Exception:
        print(f"\n!!! FAILED at cell {i}\n")
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 62)
print("PERSIST SMOKE TEST PASSED")
print("=" * 62)
