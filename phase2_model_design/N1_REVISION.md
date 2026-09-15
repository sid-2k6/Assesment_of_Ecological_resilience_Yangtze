# N1 Revision — evidence-based redesign of the Disturbance-Response Decoder

**Status: proposed revision to the approved PERSIST design.** Grounded in a monthly impulse-response experiment, not theory.

---

## Why a revision was needed

Phase 3c surfaced two problems that threatened N1, the core novelty:

- **Problem A** — drought→vegetation coupling had the *wrong sign* (`dry_z → kNDVI = +0.205`, i.e. drier→greener), because YREB summer vegetation is light-limited rather than water-limited.
- **Problem B** — seasonal z-scored anomalies carried almost no carry-over (`kNDVI corr(JJA,SON) = +0.020`), so there was no recovery trajectory to learn.

Rather than patch the design on intuition, we ran the experiment: extract **monthly** state (2010–2014, includes the 2011 drought and 2013 record heatwave) and estimate an impulse response function.

---

## Result 1 — monthly resolution recovers the persistence signal

Lagged within-county autocorrelation of anomalies:

| lag (months) | kNDVI_z | LST_z |
|---|---|---|
| 1 | **+0.209** | +0.155 |
| 2 | +0.150 | +0.128 |
| 3 | +0.049 | +0.100 |
| 4 | −0.005 | −0.036 |
| 6 | −0.010 | +0.030 |

**Seasonal baseline was kNDVI +0.020.** Monthly lag-1 is **+0.209 — a tenfold improvement.**

The hypothesis was correct: recovery completes *inside* a season, and seasonal averaging integrated it away. Memory decays to zero by roughly lag 4, i.e. a ~3-month ecological memory — precisely the structure N1's recovery term parameterises.

**This strengthens N5 considerably.** Seasonal resolution is not merely preferable to annual; it is itself too coarse for the response channel, which must be monthly.

## Result 2 — recovery IS identifiable, on LST, conditional on real shocks

Mean state anomaly following large summer heat shocks (`heat_z > 1.5`, JJA, n = 364):

| Channel | t+0 | t+1 | t+2 | t+3 | t+4 |
|---|---|---|---|---|---|
| **LST** | **+0.943** | **+0.609** | **+0.467** | **+0.281** | −0.019 |
| kNDVI | +0.184 | −0.234 | −0.194 | +0.400 | −0.324 |

LST decay ratios: **t+1/t0 = 0.646, t+2/t0 = 0.495.** This is a clean, monotonic geometric decay returning to baseline by t+4 — a textbook impulse-response curve, with an e-folding time of roughly **2.3 months**.

kNDVI, by contrast, flips sign repeatedly and shows no coherent structure.

## Result 3 — an important nuance: not all forcing→response pairs are meaningful

Continuous-forcing IRFs:

| Pair | t+0 | t+1 | t+2 | t+3 | decay ratio | verdict |
|---|---|---|---|---|---|---|
| heat_z → LST | +0.645 | +0.064 | +0.091 | +0.081 | 0.099 | **spike, not decay** |
| **dry_z → LST** | +0.381 | +0.157 | +0.106 | +0.032 | **0.412** | **genuine decay** |
| heat_z → kNDVI | +0.236 | −0.007 | −0.027 | +0.011 | −0.028 | no structure |
| dry_z → kNDVI | +0.097 | −0.068 | −0.001 | +0.033 | −0.704 | no structure |

`heat_z → LST` has a strong contemporaneous correlation (0.645) that collapses immediately. That is **near-mechanical, not ecological** — air temperature and land surface temperature are largely the same signal measured twice, and next month's temperature is independent of this month's.

`dry_z → LST` is the physically interesting pair: a water deficit raises surface temperature and that elevation **persists for 2–3 months** through soil-moisture memory and reduced evaporative cooling. That is a genuine land-surface process with real memory, and it is exactly what a resilience model should be learning.

**Design lesson:** a high contemporaneous correlation is not evidence of an identifiable response. The lag structure is what matters.

---

## The revised N1

| Element | Original | **Revised** | Justification |
|---|---|---|---|
| **Response channel** | kNDVI (vegetation) | **LST (land-surface thermal state)** | Only channel with coherent shock response (clean decay vs sign-flipping noise) |
| **Forcing axis** | drought (moisture deficit) | **water deficit + discrete heat/flood events** | Problem A: vegetation is light-limited; and `dry_z→LST` is the pair with real lag structure |
| **Temporal resolution** | seasonal | **monthly** | Result 1: lag-1 persistence 0.020 → 0.209 |
| **Estimation** | continuous regression | **event-conditioned** on large shocks | Result 3: continuous forcing is noise-dominated; resilience is defined by response to *real* disturbance |
| **Resistance** | `(∂response/∂forcing)⁻¹` | inverse of response amplitude at **t+0** | Directly measurable (+0.943 mean) |
| **Recovery** | learned decay rate | **fitted geometric decay over t+1…t+4** | Observed ratios 0.646, 0.495 — a genuine decay to fit |
| **Adaptability** | transfer-function drift | drift in fitted decay rate across years | Unchanged in spirit, now on an identifiable quantity |

### Why LST is a defensible response channel, not a downgrade

LST is not merely "temperature". It integrates soil moisture, vegetation cover, and evaporative cooling capacity — it is a genuine **land-surface state variable**. A county that heats sharply and stays hot after a shock has lost evaporative buffering capacity; one that returns quickly retains it. That is a direct, physical expression of ecological resilience, and arguably a cleaner one than greenness.

### What happens to the vegetation indicators

kNDVI, EVI, NPP and ET are **retained as state indicators** feeding the N4 self-supervised reconstruction and the composite index. They are simply **removed from the recovery-estimation channel**, where they carry no usable signal at county-month scale. Nothing is discarded; one variable is reassigned.

---

## Effect on the other novelties

| Novelty | Effect |
|---|---|
| **N1** | Revised as above. **Now empirically identifiable rather than hoped-for.** |
| **N5** | **Strengthened.** Two-level hierarchy becomes monthly → annual, with the monthly level now empirically mandatory. |
| N2, N3, N4, N6 | Unaffected. |

Sample size improves again: monthly resolution gives **1,068 × 21 × 12 = 269,136 observations**, up from 89,712 seasonal and 22,428 annual. The sample-size risk from the design's §7 is now largely retired.

---

## Honest limitations

1. The experiment used **5 years (2010–2014)**, so the monthly climatology rests on 5 samples per county-month and anomalies are noisier than a full-period fit would give. Direction and magnitude are trustworthy; exact decay coefficients are not final.
2. n = 364 large summer heat shocks is adequate for a mean profile but thin for per-county estimation. The full 21-year extraction will multiply this roughly fourfold.
3. The `t+3 = +0.400` value in the kNDVI row is almost certainly noise, not a rebound. It is reported rather than hidden precisely because it illustrates why that channel was rejected.
4. LST is sensitive to cloud-cover sampling bias: clear-sky-only retrieval means hot, dry months are over-represented. This must be checked before final use.

---

## Data requirement created by this revision

**Monthly LST and NDVI for all of 2000–2020** — currently only 2010–2014 exists.

| Item | Cost |
|---|---|
| Monthly LST, 2000–2020 | ~135 s/yr × 21 ≈ 47 min |
| Monthly NDVI, 2000–2020 | ~137 s/yr × 21 ≈ 48 min |
| Monthly forcing, 2000–2020 | ~10 min |

This supersedes the seasonal panel as the primary modelling input. The seasonal panel remains valid for the annual/inter-annual level of N5 and for reporting.
