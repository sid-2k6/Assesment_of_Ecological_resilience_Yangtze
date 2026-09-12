# Comprehensive Assessment of Ecological Resilience in the Yangtze River Economic Belt Based on Remote Sensing and Deep Learning

Research project repository.

## Deliverable in this commit

**`Literature_Survey_Ecological_Resilience_YREB.docx`** — analytical literature survey covering 42 papers, all published 2025–2026, in Harvard (Google Scholar) citation style.

Each paper is treated in a single paragraph following the structure: contribution → demonstrated strength → limitation (`However, …`) → derived research gap (`Therefore, a research gap remains in …`). A five-paragraph synthesis groups the corpus into four strands and states the consolidated gap.

## Citation integrity

No reference in this document was written from model memory. The pipeline was:

1. Candidates discovered by web search across 10 query families.
2. Every DOI resolved against the **Crossref REST API** (publisher-deposited metadata). Title, journal, volume/issue, article number, year, and full author list come from that response.
3. **42/42 verified. 0 failures. 0 pre-2025 papers.**
4. Harvard reference strings generated programmatically from the verified metadata.
5. The finished `.docx` audited for sequential inline numbering and inline-author/year agreement with Crossref — all checks pass.

Google Scholar has no public API and blocks automated access, so it cannot be queried programmatically. Crossref is the DOI registration authority that Scholar indexes from, and is therefore a stronger verification source. Every paper is reachable at its `doi.org` link.

## Provenance and reproducibility — `literature_survey/provenance/`

| File | Purpose |
|---|---|
| `verified.json` | Raw Crossref metadata for all 42 papers (audit trail) |
| `harvard.json` | Generated Harvard reference strings |
| `references.bib` | Publisher-supplied BibTeX, de-duplicated keys, for the eventual paper |
| `dois.txt` | The 42 DOIs |
| `verify_crossref.py` | Re-runs verification: `python3 verify_crossref.py dois.txt` |
| `make_harvard.py` | Builds Harvard references from Crossref |
| `make_bib.py` | Fetches BibTeX via DOI content negotiation |
| `build_docx.py` | Generates the Word document |
| `check_docx.py` | Audits numbering and author/year consistency |

Requires `python-docx`.

## Project phases

- [x] **1. Literature survey** — 42 verified papers, ranked by relatedness
- [ ] **2. Research gap analysis and novel model proposal** (≥4 novelties)
- [ ] **3. Dataset collection, EDA, preprocessing**
- [ ] **4. Three baseline models**
- [ ] **5. Proposed model with ablations**
- [ ] **6. Q1-equivalent paper**

## Consolidated research gap (from Phase 1)

Papers were scored on four axes — study area (YREB), target (ecological *resilience*), remote sensing measurement, and deep learning method. **No located study satisfies more than three simultaneously.** Belt-scale resilience work is statistical; deep architectures are pointed at land use, ecological quality, or NDVI; the nonlinearity justifying deep modelling is documented but only diagnosed post hoc.

Four compounding gaps:

1. **Spatial autocorrelation is confirmed but never modelled** — no study builds a relational graph over administrative units, despite the means being demonstrated in the same corpus.
2. **Indicator weighting is analyst-specified** throughout (entropy, AHP, PCA, TOPSIS); only one paper stress-tests it, only one reduces indicators objectively.
3. **Internal heterogeneity is ignored** — karst terrain is shown to require its own index formulation, yet the extensively karstic upstream of the belt is assessed with the same uniform index as the alluvial downstream.
4. **Terrestrial and aquatic resilience are modelled separately and their gradients disagree** — east-over-west for terrestrial versus downstream–upstream–midstream for water resilience, in the same study area.
