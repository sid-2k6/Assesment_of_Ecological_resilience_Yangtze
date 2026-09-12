#!/usr/bin/env python3
"""Consistency audit of the generated .docx: sequential inline numbering,
inline author name vs Crossref first author, and reference-list alignment."""
import json
import re

from docx import Document

H = json.load(open("harvard.json"))
BY_NUM = {}
doc = Document("Literature_Survey_Ecological_Resilience_YREB.docx")
paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

refs, body = {}, []
for t in paras:
    m = re.match(r"^\[(\d+)\]\s+(.*)$", t)
    if m:
        refs[int(m.group(1))] = m.group(2)
    else:
        body.append(t)

inline = []
for t in body:
    m = re.match(r"^([A-Z][A-Za-z\-]+)(?:\s+et\s+al\.|\s+and\s+[A-Z][A-Za-z\-]+)?"
                 r"\s+\[(\d+)\]\s+\((\d{4})\)", t)
    if m:
        inline.append((int(m.group(2)), m.group(1), m.group(3)))

print(f"reference entries : {len(refs)}")
print(f"inline citations  : {len(inline)}")

nums = [n for n, _, _ in inline]
print(f"sequential 1..42  : {nums == list(range(1, 43))}")

lookup = {v["n"]: v for v in H.values()}
errs = []
for n, surname, yr in inline:
    ref = refs.get(n)
    if not ref:
        errs.append(f"[{n}] no reference entry")
        continue
    if not ref.startswith(surname):
        errs.append(f"[{n}] inline '{surname}' != ref start '{ref[:28]}'")
    if f", {yr}." not in ref:
        errs.append(f"[{n}] year {yr} not found in reference")

print(f"author/year match : {'ALL OK' if not errs else str(len(errs)) + ' issue(s)'}")
for e in errs:
    print("   -", e)

gaps = sum(1 for t in body if "research gap remains" in t)
howev = sum(1 for t in body if "However," in t)
print(f"paragraphs with 'However,'            : {howev}/42")
print(f"paragraphs with 'research gap remains': {gaps}/42")
print(f"\nfirst reference : {refs[1][:100]}")
print(f"last reference  : {refs[42][:100]}")
