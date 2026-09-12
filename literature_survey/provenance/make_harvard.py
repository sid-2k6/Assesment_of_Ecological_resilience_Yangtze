#!/usr/bin/env python3
"""Build Google-Scholar-style Harvard reference strings from the verified
Crossref records. Re-queries Crossref for FULL author lists (verified.json
stores an abbreviated form)."""
import json
import re
import time
import urllib.parse
import urllib.request

MAILTO = "lit-survey@example.org"


def initials(given):
    if not given:
        return ""
    toks = [t for t in re.split(r"[\s\-\.]+", given) if t]
    return "".join(f"{t[0].upper()}." for t in toks)


def fmt_authors(authors):
    """Scholar Harvard: 'Li, R., Tang, X. and Liu, Y.'"""
    names = []
    for a in authors:
        fam = (a.get("family") or "").strip()
        if not fam:
            continue
        ini = initials(a.get("given", ""))
        names.append(f"{fam}, {ini}" if ini else fam)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def pages(msg):
    p = msg.get("page")
    if p:
        return f"pp.{p}" if "-" in p else f"p.{p}"
    art = msg.get("article-number")
    return f"p.{art}" if art else None


def year_of(msg):
    for k in ("published-print", "published-online", "published", "issued"):
        dp = msg.get(k, {}).get("date-parts", [[None]])
        if dp and dp[0] and dp[0][0]:
            return dp[0][0]
    return None


def harvard(msg):
    au = fmt_authors(msg.get("author", []) or [])
    yr = year_of(msg)
    title = (msg.get("title") or ["?"])[0]
    title = re.sub(r"\s+", " ", title).strip().rstrip(".")
    ct = msg.get("container-title") or []
    jour = ct[0] if ct else ""
    vol = msg.get("volume")
    iss = msg.get("issue")
    pg = pages(msg)

    bits = f"{au}, {yr}. {title}. {jour}"
    if vol:
        bits += f", {vol}"
        if iss:
            bits += f"({iss})"
    if pg:
        bits += f", {pg}"
    return bits.rstrip(".") + "."


def get(doi):
    req = urllib.request.Request(
        "https://api.crossref.org/works/" + urllib.parse.quote(doi),
        headers={"User-Agent": f"lit-survey/1.0 (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))["message"]


recs = json.load(open("verified.json"))
out = {}
for i, r in enumerate(recs, 1):
    for attempt in range(3):
        try:
            msg = get(r["doi"])
            out[r["doi"]] = {
                "n": i,
                "harvard": harvard(msg),
                "authors_inline": fmt_authors(msg.get("author", []) or []),
                "first_family": (msg.get("author", [{}])[0].get("family")
                                 if msg.get("author") else "?"),
                "n_authors": len(msg.get("author") or []),
                "year": year_of(msg),
                "doi": r["doi"],
            }
            break
        except Exception as e:
            if attempt == 2:
                print(f"FAIL {r['doi']}: {e}")
            time.sleep(2)
    time.sleep(0.3)

json.dump(out, open("harvard.json", "w"), indent=2)
for v in sorted(out.values(), key=lambda x: x["n"]):
    print(f"[{v['n']}] {v['harvard']}")
print(f"\n{len(out)} references built")
