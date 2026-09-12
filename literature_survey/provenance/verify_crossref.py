#!/usr/bin/env python3
"""Verify candidate papers exist via the Crossref REST API (authoritative
publisher-deposited metadata). Emits verified.json + a console table."""
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

MAILTO = "lit-survey@example.org"          # Crossref polite-pool identification
BASE = "https://api.crossref.org/works/"


def get(url, tries=3):
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": f"lit-survey/1.0 (mailto:{MAILTO})"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except Exception as e:
            last = e
            time.sleep(1.5 * (a + 1))
    raise last


def year_of(msg):
    for k in ("published-print", "published-online", "published", "issued",
              "created"):
        p = msg.get(k, {}).get("date-parts", [[None]])
        if p and p[0] and p[0][0]:
            return p[0][0]
    return None


def authors(msg):
    out = []
    for a in msg.get("author", []) or []:
        fam = a.get("family", "")
        given = a.get("given", "")
        if fam:
            out.append(f"{fam}, {given[:1]}." if given else fam)
    return out


def probe(doi):
    rec = {"doi": doi, "verified": False}
    try:
        msg = get(BASE + urllib.parse.quote(doi))["message"]
    except urllib.error.HTTPError as e:
        rec["error"] = f"Crossref HTTP {e.code} (DOI not registered?)"
        return rec
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    ct = msg.get("container-title") or []
    rec.update({
        "verified": True,
        "title": (msg.get("title") or ["?"])[0],
        "journal": ct[0] if ct else None,
        "publisher": msg.get("publisher"),
        "year": year_of(msg),
        "volume": msg.get("volume"),
        "issue": msg.get("issue"),
        "pages": msg.get("page") or msg.get("article-number"),
        "type": msg.get("type"),
        "authors": authors(msg),
        "n_authors": len(authors(msg)),
        "cited_by": msg.get("is-referenced-by-count"),
        "url": msg.get("URL") or f"https://doi.org/{doi}",
        "issn": (msg.get("ISSN") or [None])[0],
    })
    return rec


def main():
    dois = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    with ThreadPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(probe, dois))
    json.dump(res, open("verified.json", "w"), indent=2)

    ok = [r for r in res if r["verified"]]
    post24 = [r for r in ok if (r.get("year") or 0) >= 2025]
    print(f"{'#':>3} {'ST':4} {'YR':4} {'CIT':>4}  {'DOI':32} JOURNAL | TITLE")
    print("-" * 128)
    for i, r in enumerate(res, 1):
        if not r["verified"]:
            print(f"{i:>3} FAIL  -    -   {r['doi']:32} {r.get('error')}")
            continue
        print(f"{i:>3} OK  {str(r['year']):4} {str(r['cited_by']):>4}  "
              f"{r['doi']:32} {str(r['journal'])[:26]} | {r['title'][:56]}")
    print("-" * 128)
    print(f"candidates={len(res)}  verified={len(ok)}  "
          f"failed={len(res)-len(ok)}  year>=2025={len(post24)}")


if __name__ == "__main__":
    main()
