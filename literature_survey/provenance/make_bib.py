#!/usr/bin/env python3
"""Fetch publisher-deposited BibTeX for every verified DOI via Crossref
content negotiation, so the reference list is guaranteed non-fabricated."""
import json
import time
import urllib.request

recs = json.load(open("verified.json"))
out = []
for i, r in enumerate(recs, 1):
    if not r.get("verified"):
        continue
    req = urllib.request.Request(
        "https://doi.org/" + r["doi"],
        headers={"Accept": "application/x-bibtex",
                 "User-Agent": "lit-survey/1.0 (mailto:lit-survey@example.org)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                bib = resp.read().decode("utf-8", "ignore").strip()
            out.append(f"% [{i}] {r['journal']} ({r['year']})\n{bib}\n")
            print(f"[{i:>2}] ok  {r['doi']}")
            break
        except Exception as e:
            if attempt == 2:
                print(f"[{i:>2}] FAIL {r['doi']}: {e}")
            time.sleep(2 * (attempt + 1))
    time.sleep(0.4)

with open("references.bib", "w") as f:
    f.write("\n".join(out))
print(f"\nwrote references.bib with {len(out)} entries")
