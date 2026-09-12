#!/usr/bin/env python3
"""Build the Word (.docx) literature survey in analytical review style with
Google-Scholar Harvard references. Reference numbering follows discussion
order; all metadata originates from verified Crossref records."""
import json

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

H = json.load(open("harvard.json"))

# Discussion order -> DOI. Ordered by relatedness tier established in Phase 1.
ORDER = [
    "10.1007/s12061-026-09943-8",   # 1
    "10.1007/s10668-026-07320-6",   # 2
    "10.3390/su18010256",           # 3
    "10.3390/rs17243941",           # 4
    "10.3390/buildings16040844",    # 5
    "10.3390/land15040657",         # 6
    "10.3390/rs17030558",           # 7
    "10.3390/rs18010031",           # 8
    "10.3390/su17188265",           # 9
    "10.3390/land15020261",         # 10
    "10.3390/ijgi15090383",         # 11
    "10.1038/s41598-026-66902-6",   # 12
    "10.1007/s11442-026-1510-0",    # 13
    "10.3390/su18179069",           # 14
    "10.3390/land15071167",         # 15
    "10.3390/rs18040643",           # 16
    "10.1007/s10708-026-11666-9",   # 17
    "10.1134/S0097807825700289",    # 18
    "10.1038/s41598-025-20575-9",   # 19
    "10.3390/su17125305",           # 20
    "10.3390/su18147183",           # 21
    "10.3389/fevo.2026.1900995",    # 22
    "10.3390/su18126322",           # 23
    "10.3390/rs18050786",           # 24
    "10.3390/land15030429",         # 25
    "10.3390/rs18173023",           # 26
    "10.1038/s41598-026-55191-8",   # 27
    "10.3389/fmars.2026.1632093",   # 28
    "10.1007/s43621-025-02317-z",   # 29
    "10.3390/math14010064",         # 30
    "10.3390/rs18020243",           # 31
    "10.3390/agriculture15131358",  # 32
    "10.1007/s10661-026-15385-z",   # 33
    "10.3390/land14030598",         # 34
    "10.3390/su17157114",           # 35
    "10.1038/s41598-025-05298-1",   # 36
    "10.3390/su17198528",           # 37
    "10.3390/land14091769",         # 38
    "10.1038/s41598-025-20425-8",   # 39
    "10.3390/su172210267",          # 40
    "10.3389/fbuil.2025.1690346",   # 41
    "10.3390/su18136410",           # 42
]
NUM = {doi: i + 1 for i, doi in enumerate(ORDER)}

TITLE = ("Literature Survey: Comprehensive Assessment of Ecological "
         "Resilience in the Yangtze River Economic Belt Based on Remote "
         "Sensing and Deep Learning")

# Each entry: (doi, body text). "{n}" is replaced with the citation number.
P = [
("10.1007/s12061-026-09943-8",
 "Li et al. [{n}] (2026) examined the spatiotemporal evolution of the ecological "
 "resilience gradient across 1,070 counties of the Yangtze River Economic Belt "
 "(YREB), using multi-source remote sensing data linked to land use information "
 "and investigating the mechanisms driving that evolution at multiple spatial "
 "scales. Their analysis established that ecological resilience across the belt "
 "remained persistently low between 2000 and 2020 and followed a modest "
 "decline-then-rebound trajectory, falling by approximately 2.17 per cent over "
 "2000-2010 before recovering by approximately 1.90 per cent over 2010-2020, "
 "and it delivered the most spatially complete county-level resilience "
 "accounting yet published for the region. However, the study constructs its "
 "resilience index and attributes its drivers entirely through conventional "
 "statistical and spatial-analytic procedures, so the nonlinear interactions "
 "among indicators and the temporal dependence between successive resilience "
 "states are not learned from the data but imposed by the aggregation scheme. "
 "Therefore, a research gap remains in modelling county-level YREB resilience "
 "with representations capable of learning nonlinear indicator interaction and "
 "long-range temporal dependence, rather than fixing them a priori through "
 "weighted aggregation."),

("10.1007/s10668-026-07320-6",
 "Fu et al. [{n}] (2026) analysed driving factors and conducted multi-scenario "
 "simulation of ecological resilience in the metropolitan area of the middle "
 "reaches of the Yangtze River, employing interpretable machine learning models "
 "to move the analysis from static assessment towards dynamic simulation. Their "
 "framework demonstrated that resilience trajectories can be projected rather "
 "than merely described, and it translated the resulting projections into "
 "planning and governance recommendations transferable to comparable "
 "metropolitan areas. However, the work is confined to a sub-region of the "
 "belt, and its interpretable machine learning models are tree-based estimators "
 "applied to tabular indicator vectors, which treat each spatial unit as an "
 "independent observation and therefore cannot exploit the spatial dependence "
 "between neighbouring units or the sequential structure of the observation "
 "record. Therefore, a research gap remains in extending scenario-based "
 "resilience simulation to the full belt using architectures that represent "
 "spatial adjacency and temporal ordering explicitly."),

("10.3390/su18010256",
 "Li et al. [{n}] (2025) investigated long-term spatiotemporal patterns and "
 "driving forces of ecological quality across the entire Yangtze River Economic "
 "Belt at basin scale, applying geographically weighted regression with "
 "regularisation to two decades of remote sensing observations reshaped by rapid "
 "urbanisation and intensive ecological restoration. Their results reported "
 "substantial overall improvement, with roughly 69.6 per cent of the belt "
 "classified as good or excellent by 2024, approximately 34.6 per cent of land "
 "improving continuously and approximately 6.4 per cent subject to persistent "
 "degradation risk, thereby supplying a spatially explicit degradation-hotspot "
 "inventory for the whole study area. However, the dependent variable is "
 "ecological quality rather than ecological resilience, so the analysis "
 "characterises the current state of the ecosystem without decomposing its "
 "capacity to resist disturbance, recover from it, or adapt to it. Therefore, a "
 "research gap remains in distinguishing ecological quality from ecological "
 "resilience at basin scale, since a region may exhibit high measured quality "
 "while retaining little capacity to absorb future disturbance."),

("10.3390/rs17243941",
 "Tong et al. [{n}] (2025) decrypted the spatiotemporal dynamics and "
 "optimisation pathways of ecological resilience under a panarchy-inspired "
 "framework in the Wuhan Metropolitan Area, conducting a nested local-global "
 "assessment for 2000-2020 and coupling XGBoost-SHAP attribution with a dynamic "
 "Bayesian network. Their approach recovered an evolving causal network of "
 "resilience rather than a static ranking of variable importance, identified "
 "forest and construction land as the pivotal drivers, and reported a slight "
 "overall resilience decline accompanied by pronounced east-west disparity. "
 "However, the analysis is restricted to a single metropolitan area, and the "
 "causal network is estimated over aggregated indicator time series rather than "
 "over the spatial field itself, so the mechanism by which resilience "
 "propagates between adjacent administrative units is not represented. "
 "Therefore, a research gap remains in scaling causal, mechanism-oriented "
 "resilience attribution from a single metropolitan area to an entire economic "
 "belt while retaining explicit representation of spatial propagation."),

("10.3390/buildings16040844",
 "Fu et al. [{n}] (2026) explored the nonlinear effects of the built "
 "environment on ecological resilience in the high-density city of Wuhan, "
 "constructing a one-kilometre grid ecological resilience index that integrates "
 "ecosystem resistance, adaptability and recovery, and applying a "
 "Bayesian-optimised XGBoost model to recover nonlinear driver responses. Their "
 "study is notable for confirming statistically significant spatial "
 "autocorrelation in the resilience index itself, establishing that resilience "
 "values at neighbouring locations are not independent, and for operationalising "
 "the three-component resilience decomposition at fine spatial granularity. "
 "However, having demonstrated that spatial autocorrelation is present, the "
 "study proceeds to fit a gradient-boosted tree ensemble that models each grid "
 "cell as an independent sample, so the confirmed spatial structure is treated "
 "as a diagnostic property of the data rather than as information the predictor "
 "can exploit. Therefore, a research gap remains in modelling ecological "
 "resilience with architectures that consume spatial dependence directly, since "
 "the presence of autocorrelation established by this work is precisely the "
 "condition under which independent-sample estimators are inefficient and their "
 "uncertainty estimates unreliable."),

("10.3390/land15040657",
 "Wei et al. [{n}] (2026) performed multi-scenario regional spatial simulation "
 "of the Yangtze River Economic Belt using a UNet++ architecture, reporting high "
 "precision in large-scale spatial forecasting with an average test accuracy of "
 "approximately 99.32 per cent and claiming effective capture of nonlinear "
 "evolutionary characteristics. Their contribution is significant for this "
 "review because it constitutes the only located study that is simultaneously "
 "conducted at full YREB extent, driven by remote sensing, and built on a deep "
 "neural architecture, thereby establishing that deep segmentation and "
 "forecasting networks can be trained successfully over the entire belt. "
 "However, the prediction target is land use and spatial pattern rather than "
 "ecological resilience, and the headline accuracy is reported as an aggregate "
 "pixel-wise figure on a task in which unchanged pixels overwhelmingly dominate "
 "the class distribution, so the metric is unlikely to reflect performance on "
 "the changed pixels that carry the ecological signal. Therefore, a research gap "
 "remains in redirecting belt-scale deep architectures from land use prediction "
 "towards ecological resilience estimation, and in evaluating them with "
 "change-sensitive and class-balanced metrics rather than aggregate accuracy."),

("10.3390/rs17030558",
 "Gong et al. [{n}] (2025) proposed a deep-learning-based remote sensing "
 "ecological index, replacing the linear principal component aggregation of the "
 "conventional remote sensing ecological index with an autoencoder neural "
 "network incorporating long short-term memory modules, and additionally "
 "folding indicators of human economic activity into the input set. Their "
 "framework demonstrated that learned nonlinear aggregation of ecological "
 "indicators yields a more faithful composite index than fixed linear weighting, "
 "and it established neural index construction as a legitimate methodological "
 "contribution in ecological remote sensing. However, the aggregation operates "
 "per observation unit over its temporal sequence and contains no spatial "
 "component, so neighbourhood context is discarded, and the target quantity is "
 "ecological quality rather than resilience, with the study conducted outside "
 "the Yangtze River Economic Belt. Therefore, a research gap remains in "
 "extending learned nonlinear index construction so that it aggregates across "
 "space as well as across time, and in retargeting it from ecological quality "
 "onto the resistance, recovery and adaptability components of resilience."),

("10.3390/rs18010031",
 "Xie et al. [{n}] (2025) evaluated ecological environmental quality from "
 "satellite observations under a pressure-state-response framework, selecting "
 "remote sensing indicators according to the established evaluation system and "
 "employing deep neural networks to quantify the resulting composite. Their "
 "study is important as a precedent for combining a theory-driven indicator "
 "framework with a learned aggregation function, in which ecological theory "
 "constrains which variables enter the model while the network determines how "
 "they combine. However, the network is a fully connected architecture applied "
 "to indicator vectors without spatial or temporal structure, so it captures "
 "nonlinearity in aggregation while remaining blind to the spatial and sequential "
 "organisation of the underlying fields, and the framework addresses quality "
 "rather than resilience. Therefore, a research gap remains in combining a "
 "theory-constrained indicator framework with a spatiotemporally structured "
 "learner, so that ecological theory governs the inputs while the architecture "
 "matches the geographic and dynamic nature of the data."),

("10.3390/su17188265",
 "Yao et al. [{n}] (2025) assessed the impact of climate change on the "
 "ecological resilience of the Yangtze River Economic Belt, evaluating "
 "resilience under four future climate scenarios and computing a climate change "
 "impact index for each. Their projections returned index values of -0.8005, "
 "-0.8924, -0.9540 and -1.2298 across the four scenarios, indicating a general "
 "decline in resilience across the belt that intensifies monotonically with "
 "forcing level, and thereby supplying a directional benchmark against which "
 "later projections can be checked. However, the resilience response to climate "
 "forcing is derived through scenario-driven computation rather than learned "
 "from observed resilience dynamics, so the study projects consequences of "
 "assumed relationships instead of estimating those relationships from the "
 "historical record. Therefore, a research gap remains in learning the "
 "climate-resilience response function directly from observed multi-decadal "
 "remote sensing series, so that scenario projection rests on empirically "
 "estimated rather than prescribed sensitivities."),

("10.3390/land15020261",
 "Chen et al. [{n}] (2026) analysed the nonlinear impact of population "
 "shrinkage on urban ecological resilience through threshold effect modelling "
 "applied to city-level panel data from the Yangtze River Economic Belt. Their "
 "findings indicated an overall upward trend in urban ecological resilience "
 "across the belt characterised by pronounced spatial disparity, with eastern "
 "cities exhibiting higher resilience than western ones, and they introduced "
 "demographic contraction as an explanatory factor that remote-sensing-centred "
 "studies routinely omit. However, the analysis operates on aggregated "
 "city-level panel indicators drawn from statistical sources without remote "
 "sensing input, so ecological resilience is inferred from socioeconomic proxies "
 "rather than measured from observed land surface condition. Therefore, a "
 "research gap remains in integrating demographic and socioeconomic drivers of "
 "this kind with pixel-level remote sensing evidence within a single model, so "
 "that human and biophysical determinants of resilience are estimated jointly "
 "rather than in separate literatures."),

("10.3390/ijgi15090383",
 "Li et al. [{n}] (2026) explored nonlinear response patterns and interaction "
 "effects of urban resilience in the Yangtze River Economic Belt using an "
 "XGBoost-SHAP model, decomposing resilience into economic, infrastructure, "
 "ecological and social dimensions. Their results showed that overall resilience "
 "rose steadily while the four dimensions followed distinct trajectories, with "
 "economic and infrastructure resilience improving most rapidly, ecological "
 "resilience increasing steadily and social resilience lagging, which "
 "establishes empirically that the ecological dimension evolves on its own path "
 "and warrants separate treatment. However, the ecological dimension is "
 "represented by a small number of aggregate statistical indicators nested "
 "within a composite urban resilience index, so its internal spatial structure "
 "and its biophysical determinants are not resolved. Therefore, a research gap "
 "remains in elevating ecological resilience from a coarse sub-dimension of "
 "composite urban resilience to a spatially resolved target estimated from "
 "biophysical observation in its own right."),

("10.1038/s41598-026-66902-6",
 "Ding et al. [{n}] (2026) constructed a county-level ecological resilience "
 "index for the Guanzhong Plain urban agglomeration over 2000-2020 under a "
 "resistance-recovery-adaptability framework, disentangling its drivers by "
 "coupling an optimised-parameter geographical detector with a Bayesian "
 "spatiotemporal model estimated through integrated nested Laplace "
 "approximation. Their analysis revealed a polarised spatial structure with "
 "persistent high-value clustering along the southern Qinling foothills and the "
 "northern Loess Plateau margin against a low-value cluster in the central Wei "
 "River valley, and it represents the methodologically strongest statistical "
 "treatment of spatial and temporal dependence located in this review. However, "
 "the study addresses a different and considerably smaller agglomeration, and "
 "its Bayesian formulation imposes a parametric dependence structure with "
 "additive covariate effects, so nonlinear interactions among drivers are "
 "admitted only where explicitly specified. Therefore, a research gap remains "
 "in retaining the rigorous spatiotemporal dependence handling of Bayesian "
 "hierarchical models while relaxing their parametric and additivity "
 "assumptions, and in demonstrating that a learned alternative outperforms them "
 "on the same county-level resilience task."),

("10.1007/s11442-026-1510-0",
 "Li et al. [{n}] (2026) developed a scenario-based modelling and driver "
 "analysis framework for ecological resilience in the Qinling-Daba Mountains, "
 "evaluating resilience trajectories under contrasting development futures. "
 "Their results revealed a significant upward resilience trend with high values "
 "in the central and southern mountains and low values in urban lowlands, "
 "identified vegetation index, slope, elevation, temperature and nighttime light "
 "as the key drivers with resilience correlating positively with high vegetation "
 "and steep slope and negatively with sparse vegetation and flat terrain, and "
 "showed resilience improving under ecological protection, plateauing under "
 "natural development and declining under a farmland-protection-first scenario. "
 "However, the study area is a mountainous region whose terrain-dominated "
 "resilience regime differs fundamentally from the mixed alluvial, urban and "
 "karst terrains composing the Yangtze River Economic Belt, and the analysis "
 "does not employ deep learning. Therefore, a research gap remains in "
 "establishing whether the driver hierarchy validated in a homogeneous "
 "mountainous setting transfers to a heterogeneous multi-terrain economic belt, "
 "and in whether a single driver hierarchy is even well defined across such "
 "internal diversity."),

("10.3390/su18179069",
 "Zhao et al. [{n}] (2026) conducted a simulation-based multi-scenario "
 "assessment of comprehensive ecological risk and resilience in the Pearl River "
 "Delta, quantifying risk through a landscape ecological risk index and a "
 "habitat degradation index while deriving resilience from an "
 "adaptability-resistance-recovery framework, and fusing both into a "
 "comprehensive ecological risk-resilience index. Their formulation demonstrated "
 "that risk and resilience are complementary rather than redundant "
 "constructs and that their joint representation supports sharper "
 "prioritisation than either alone. However, the two constructs are combined "
 "through index arithmetic after independent computation rather than estimated "
 "jointly, so shared information between risk and resilience is not exploited "
 "during estimation, and the study addresses a different delta system. "
 "Therefore, a research gap remains in estimating ecological risk and "
 "resilience as coupled outputs of a single model, in which shared "
 "representation learning allows each target to regularise the other."),

("10.3390/land15071167",
 "Qu et al. [{n}] (2026) applied interpretable machine learning to diagnose "
 "ecological quality dynamics derived from the remote sensing ecological index "
 "across the Yangtze River Delta, standardising the annual index by year to "
 "capture relative trajectories rather than absolute change under a fixed "
 "loading system. Their central empirical result is that the regional mean index "
 "moved only marginally, from 0.639 in 2000 to 0.632 in 2025, while fluctuating "
 "markedly throughout the period, from which they conclude that long-term "
 "ecological change in the region is nonlinear. However, the study diagnoses "
 "ecological quality rather than resilience and employs post hoc attribution "
 "over tree ensembles, so the demonstrated nonlinearity is characterised but not "
 "captured by a model able to represent it generatively. Therefore, a research "
 "gap remains in adopting model classes whose inductive bias matches the "
 "nonlinear, fluctuation-dominated behaviour this study documents, since a "
 "near-stationary long-run mean concealing large interannual variance is "
 "precisely the regime in which linear trend estimation discards the signal of "
 "interest."),

("10.3390/rs18040643",
 "Yang et al. [{n}] (2026) combined large language models with satellite "
 "embeddings to evaluate the ecological quality of the Tibetan Plateau from 2000 "
 "to 2024, motivating their design by the limitations of assessments that rely "
 "on medium-resolution single-source optical imagery, privilege natural factors "
 "while neglecting human impact, and struggle with temporal interpretability and "
 "continuity in complex terrain. Their model achieved high predictive accuracy "
 "against field observations for key ecological indicators, with reported "
 "coefficients of determination of 0.9923 for the fraction of absorbed "
 "photosynthetically active radiation and 0.8690 for above-ground biomass, and "
 "an interpretable conditional random field layer partitioned the plateau into "
 "potential risk, enhancement potential and stable conservation management "
 "zones. However, the framework targets ecological quality in a high-altitude "
 "plateau environment rather than resilience in an economically intensive river "
 "corridor, and the reported validation coefficients are obtained against "
 "sparse point observations whose spatial representativeness across such terrain "
 "is difficult to establish. Therefore, a research gap remains in transferring "
 "foundation-model embedding representations and interpretable zoning outputs "
 "from plateau ecological quality assessment to resilience assessment in "
 "densely developed basins, and in validating them against spatially "
 "distributed rather than point reference data."),

("10.1007/s10708-026-11666-9",
 "Yang [{n}] (2026) assessed tourism ecological environment quality using a "
 "framework combining a ResNeXt backbone, a YOLOv5s detection stage and a long "
 "short-term memory temporal module, integrating remote sensing imagery with "
 "spatial analysis. The study reported that coupling remote sensing, spatial "
 "analysis and deep learning improves the reliability and timeliness of "
 "ecological environment assessment relative to conventional procedures, and it "
 "provides a practical monitoring workflow oriented towards management "
 "decisions. However, the composition of the pipeline is loosely motivated, "
 "since object detection addresses the localisation of discrete bounded entities "
 "whereas regional ecological assessment concerns continuous surface fields, and "
 "the study neither ablates the contribution of each stage nor targets "
 "resilience. Therefore, a research gap remains in designing deep architectures "
 "for ecological assessment whose components are justified by the structure of "
 "the estimation problem and whose individual contributions are demonstrated "
 "through systematic ablation."),

("10.1134/S0097807825700289",
 "Tang et al. [{n}] (2025) examined the spatial-temporal evolution of water "
 "ecological resilience in the Yangtze River Economic Belt together with the "
 "non-stationarity of its influencing factors. Their findings established that "
 "water ecological resilience across the belt remained low while following a "
 "fluctuating upward trend, and, critically for the present review, that its "
 "regional ordering follows a downstream, then upstream, then midstream "
 "sequence, which contradicts the monotonic east-over-west gradient reported for "
 "terrestrial resilience in the same study area. However, the analysis proceeds "
 "through statistical indicator systems and spatially varying regression without "
 "remote sensing input, and it treats the aquatic subsystem in isolation from "
 "the terrestrial one. Therefore, a research gap remains in reconciling the "
 "divergent aquatic and terrestrial resilience gradients of the belt within a "
 "single model, since their disagreement indicates that a resilience assessment "
 "restricted to terrestrial indicators systematically misrepresents a river "
 "corridor whose defining feature is its hydrological network."),

("10.1038/s41598-025-20575-9",
 "Zhong et al. [{n}] (2025) analysed regional differences, dynamic evolution "
 "and driving factors of ecological resilience across China's urban "
 "agglomerations, comparing resilience levels and trajectories between "
 "agglomerations. They reported that internet penetration, level of "
 "informatisation and innovation capacity exert statistically significant but "
 "spatially heterogeneous effects on ecological resilience, and concluded that "
 "spatial co-governance and zoning management are necessary for regional "
 "ecological security. However, the study operates on aggregated statistical "
 "indicators at agglomeration level without remote sensing observation, and the "
 "heterogeneity of driver effects is reported without being modelled, so the "
 "spatial variation in driver influence is documented rather than predicted. "
 "Therefore, a research gap remains in models that represent spatially varying "
 "driver effects endogenously, so that heterogeneity in how determinants act "
 "upon resilience is an estimated property of the model rather than a "
 "post hoc observation."),

("10.3390/su17125305",
 "Li et al. [{n}] (2025) assessed and simulated urban ecosystem resilience in "
 "the Jinan metropolitan area by coupling a resistance-adaptability-recovery "
 "index with Markov-FLUS land use simulation. Their results described a "
 "fluctuating resilience trajectory accompanied by functional intensification "
 "in high-value areas and escalating vulnerability in low-value areas, together "
 "with a spatial dichotomy between resilient southern mountains and less "
 "resilient northern plains dominated by natural factors, thereby establishing "
 "the standard template for coupling a resilience index with a cellular land use "
 "simulator. However, Markov-FLUS propagates land use states through transition "
 "probabilities and neighbourhood suitability rules that are calibrated rather "
 "than learned, so the simulator cannot represent regime shifts absent from the "
 "calibration period. Therefore, a research gap remains in replacing "
 "rule-calibrated cellular simulation with learned spatiotemporal dynamics for "
 "resilience projection, and in benchmarking the learned alternative directly "
 "against the Markov-FLUS convention it seeks to displace."),

("10.3390/su18147183",
 "Yue et al. [{n}] (2026) derived socio-ecological resilience pathways for the "
 "resource-exhausted city of Jiaozuo using interpretable machine learning, "
 "conducting scenario simulations to compare intervention strategies. Their "
 "analysis found that a green transformation pathway produced the strongest "
 "model-predicted gains and, importantly, verified that this pathway remained "
 "the highest-ranked option under alternative subsystem-weighting schemes, "
 "thereby subjecting the conclusion to a robustness test that the resilience "
 "literature rarely applies. However, the study addresses a single resource-"
 "dependent city and interrogates the sensitivity of conclusions to weighting "
 "without removing the underlying dependence on analyst-specified weights. "
 "Therefore, a research gap remains in eliminating rather than merely stress-"
 "testing subjective indicator weighting, by learning the aggregation of "
 "resilience components from data under supervision or reconstruction "
 "objectives, while retaining the robustness-checking discipline this study "
 "demonstrates."),

("10.3389/fevo.2026.1900995",
 "Wang et al. [{n}] (2026) analysed the spatiotemporal evolutionary "
 "characteristics of rural digitalisation and ecosystem services at county level "
 "across the Yangtze River economic belt. Their results indicated that rural "
 "digitalisation exhibits a gradient pattern with the eastern region leading and "
 "the western region catching up, while ecosystem services display spatial "
 "differentiation decreasing from downstream to upstream, providing county-level "
 "evidence at precisely the administrative granularity relevant to resilience "
 "governance. However, ecosystem services are quantified through value transfer "
 "and coefficient-based accounting rather than direct biophysical observation, "
 "and resilience is not addressed. Therefore, a research gap remains in "
 "estimating county-level ecological condition in the belt from observed surface "
 "biophysical state rather than from land-use-mediated valuation coefficients, "
 "whose transfer across the belt's diverse ecological contexts is itself a "
 "source of unquantified error."),

("10.3390/su18126322",
 "Peng et al. [{n}] (2026) examined how digital technological innovation "
 "influences the coordination between urban renewal and ecological resilience, "
 "using data from 108 cities of the Yangtze River Economic Belt over 2012-2023 "
 "and constructing an urban renewal indicator system spanning infrastructure "
 "construction, social function development, and cultural and leisure "
 "facilities. The study provides a recent, belt-wide socioeconomic panel and "
 "articulates a mechanism linking technological capability to the "
 "reconciliation of redevelopment pressure with ecological capacity. However, "
 "the analysis is econometric and city-aggregated, so ecological resilience "
 "enters as a composite index computed from statistical yearbook variables "
 "rather than as an observed spatial field. Therefore, a research gap remains in "
 "coupling belt-wide socioeconomic panels of this kind to spatially explicit "
 "remote sensing measurement, so that the ecological side of the relationship is "
 "observed rather than proxied."),

("10.3390/rs18050786",
 "Wang et al. [{n}] (2026) assessed the health of terrestrial ecosystems across "
 "China using an adaptive indicator reduction method, proposing a scalable and "
 "objective methodology intended to support national ecological monitoring, "
 "zoning and policy evaluation across regions of differing character. Their "
 "framework delivered a unified yet adaptable index system in which indicator "
 "selection responds to regional conditions rather than being fixed in advance, "
 "which directly addresses the redundancy and arbitrariness endemic to "
 "expert-specified indicator sets. However, the adaptation concerns which "
 "indicators are retained rather than how they are nonlinearly combined, the "
 "target is ecosystem health rather than resilience, and the reduction operates "
 "as a preprocessing stage decoupled from any downstream predictive model. "
 "Therefore, a research gap remains in unifying adaptive indicator selection "
 "with learned nonlinear aggregation inside a single end-to-end model, so that "
 "which variables matter and how they interact are determined jointly rather "
 "than sequentially."),

("10.3390/land15030429",
 "Cheng et al. [{n}] (2026) developed a vitality-organisation-resilience "
 "framework augmented with ecosystem services and an enhanced ecological quality "
 "indicator to assess ecosystem health in the Henan section of the Yellow River "
 "Basin from 2000 to 2020, applying XGBoost-SHAP attribution to identify "
 "nonlinear drivers. Their analysis identified a persistent high-west, low-east "
 "health gradient with overall decline, with western mountains remaining healthy "
 "while eastern plains, urban and intensively cultivated areas degraded, and it "
 "recovered explicit threshold effects at which driver influence changes "
 "character. However, resilience appears as one sub-component nested within a "
 "composite health index rather than as the estimation target, and threshold "
 "effects are recovered post hoc from a tree ensemble rather than represented "
 "within the model. Therefore, a research gap remains in treating resilience as "
 "the primary target rather than a subordinate component of ecosystem health, "
 "while preserving the capacity to detect the driver thresholds this study shows "
 "to be ecologically consequential."),

("10.3390/rs18173023",
 "Ke et al. [{n}] (2026) proposed GT-LandSDS, a spatiotemporal framework for "
 "land use simulation coupling cellular automata with a graph attention network "
 "and a transformer, in which the graph attention component captures "
 "higher-order spatial dependencies among land parcels, transformer "
 "self-attention extracts change characteristics from multi-period observations, "
 "and an agent-based component represents traffic, resident and government "
 "decision behaviour. Architecturally this is the closest located analogue to "
 "the design required for spatially and temporally structured regional "
 "modelling, demonstrating that graph attention and self-attention can be "
 "combined coherently over geographic units at regional scale. However, the "
 "framework simulates land use transition rather than estimating ecological "
 "resilience, and its graph is constructed over land parcels using spatial "
 "proximity rather than over administrative units using ecological or "
 "hydrological connectivity. Therefore, a research gap remains in adapting "
 "coupled graph-attention and transformer architectures from land use "
 "transition simulation to ecological resilience estimation, with graph "
 "construction reflecting ecological and hydrological connectivity rather than "
 "geometric adjacency alone."),

("10.1038/s41598-026-55191-8",
 "Soula et al. [{n}] (2026) proposed a BiLSTM-CNN model for predicting "
 "large-scale temporal-spatial dynamics of the normalised difference vegetation "
 "index, describing it as a composite progressive processing architecture able "
 "to investigate vegetation trends that may be abrupt or barely perceptible, "
 "localised or extensive, and unfolding over short or long timescales. This "
 "explicit design objective of detecting disturbance across heterogeneous "
 "spatial extents and temporal scales corresponds closely to the multi-scale "
 "disturbance-and-recovery signature that defines ecological resilience. "
 "However, the model predicts a single vegetation index rather than a composite "
 "resilience construct, and it therefore captures the vegetation response to "
 "disturbance without representing the resistance and adaptive capacity "
 "components that distinguish resilience from greenness dynamics. Therefore, a "
 "research gap remains in generalising multi-scale bidirectional recurrent-"
 "convolutional prediction from a univariate vegetation index to a multivariate "
 "resilience target, in which recovery behaviour must be estimated jointly with "
 "resistance and adaptability."),

("10.3389/fmars.2026.1632093",
 "Yuan et al. [{n}] (2026) developed a two-stage deep learning framework for "
 "mangrove change prediction, in which an enhanced U-Net incorporating "
 "squeeze-and-excitation and convolutional block attention modules extracts "
 "annual masks from multi-temporal Landsat imagery spanning 1993 to 2023, "
 "achieving an intersection over union of 0.815 and an F1 score of 0.928, after "
 "which a U-Net-ConvLSTM performs spatiotemporal forecasting. Of particular "
 "methodological interest, they introduce an optional asymmetric ecological "
 "constraint loss that functions primarily as a safeguard against "
 "ecologically implausible long-term runaway trends, yielding modest accuracy "
 "gains while enforcing domain plausibility. However, the ecological constraint "
 "encodes assumptions specific to mangrove succession in a small coastal study "
 "site, and the framework predicts land cover extent rather than a resilience "
 "index. Therefore, a research gap remains in formulating domain-constrained "
 "loss functions that encode resilience-specific ecological plausibility at "
 "regional scale, since the principle of penalising implausible trajectories "
 "demonstrated here has not been applied to composite resilience targets."),

("10.1007/s43621-025-02317-z",
 "Kaur and Sharma [{n}] (2025) reviewed multimodal graph neural networks for "
 "earth observation and sustainable resource management, surveying how "
 "heterogeneous inputs including optical and synthetic aperture radar imagery, "
 "in-situ sensor readings, geospatial vector layers and socio-economic records "
 "can be integrated into unified relational representations, and setting out a "
 "research roadmap. Their synthesis provides the theoretical justification for "
 "representing geographic units as nodes in a heterogeneous graph carrying "
 "attributes of fundamentally different provenance, which is precisely the "
 "integration problem posed by combining satellite indicators with statistical "
 "yearbook variables. However, the contribution is a review that identifies "
 "capability and prescribes direction without instantiating or evaluating a "
 "model on any ecological assessment task. Therefore, a research gap remains in "
 "empirically demonstrating that multimodal graph representations improve "
 "ecological resilience estimation over independent-unit baselines, since the "
 "architectural rationale has been articulated in the abstract but not "
 "validated in this application domain."),

("10.3390/math14010064",
 "Aldossary [{n}] (2025) proposed GTNet, a graph-transformer neural network for "
 "ecological health monitoring in smart cities, presented as a data-driven "
 "predictive framework providing real-time estimation of urban garden health and "
 "addressing the absence of proactive environmental monitoring. The work is "
 "relevant as evidence that hybrid graph and transformer architectures function "
 "on ecological rather than purely computer-vision targets, extending their "
 "demonstrated applicability into environmental assessment. However, the spatial "
 "extent is confined to intra-urban vegetated parcels monitored at fine temporal "
 "resolution, which is a fundamentally different estimation regime from "
 "annual-resolution assessment across a continental-scale economic corridor, and "
 "the ecological target is a health proxy rather than a resilience construct. "
 "Therefore, a research gap remains in establishing whether graph-transformer "
 "hybrids retain their advantage when transferred from dense intra-urban "
 "monitoring to sparse, annually sampled regional resilience assessment over "
 "administrative units."),

("10.3390/rs18020243",
 "Li et al. [{n}] (2026) surveyed Mamba-based architectures, hybrid paradigms "
 "and future directions for remote sensing, examining visual state-space models "
 "that offer linear-time sequence processing with selective recurrence and "
 "analysing the extent to which this efficiency promise is realised in practice. "
 "Their critical posture is valuable because it interrogates rather than "
 "advertises the claimed advantages, providing an evidence-based basis for "
 "deciding whether state-space sequence modelling is warranted for long "
 "geospatial time series. However, the survey addresses architectural efficiency "
 "and representational capacity in generic remote sensing tasks, predominantly "
 "classification, segmentation and detection, and does not consider composite "
 "ecological index estimation or resilience assessment. Therefore, a research "
 "gap remains in determining whether linear-complexity state-space sequence "
 "models confer measurable benefit for multi-decadal ecological resilience "
 "series, whose sequence lengths are short enough that the efficiency advantage "
 "motivating these architectures may not materialise."),

("10.3390/agriculture15131358",
 "Guo et al. [{n}] (2025) coupled assessment of land use change and ecological "
 "benefits using multi-source remote sensing data for the urban agglomeration "
 "in the middle reaches of the Yangtze River, developing a four-quadrant and "
 "coupling-degree framework relating the remote sensing ecological index to an "
 "ecological service index and applying Geodetector analysis to identify "
 "influencing factors and their interactions. Their results quantified a "
 "reconfiguration of the ecological pattern in which the jointly high-performing "
 "quadrant contracted by 13,800 square kilometres while quadrants exhibiting "
 "divergence between the two indices expanded, demonstrating that ecological "
 "quality and ecological service provision can decouple spatially. However, the "
 "coupling is diagnosed through quadrant classification of independently "
 "computed indices rather than modelled, and resilience is not among the "
 "quantities assessed. Therefore, a research gap remains in modelling the joint "
 "distribution of ecological state and ecological function rather than "
 "cross-tabulating separately derived indices, particularly because the "
 "decoupling this study documents implies that neither index alone suffices to "
 "characterise ecological condition."),

("10.1007/s10661-026-15385-z",
 "Yu et al. [{n}] (2026) investigated whether the ecological quality of "
 "river-connected lake basins exceeds that of disconnected basins in the lower "
 "reaches of the Yangtze River, comparing remote-sensing-derived ecological "
 "quality between basin types over 2015 to 2023. They found low ecological "
 "quality in connected lake basins together with dense cold-spot clustering of "
 "the ecological index around lake areas, and concluded that restoration effort "
 "should be prioritised in connected basins, thereby supplying a specific and "
 "spatially testable pattern within the lower belt. However, the study is "
 "confined to lake basins of the lower reaches and to ecological quality rather "
 "than resilience, and hydrological connectivity is treated as a categorical "
 "grouping variable rather than as a continuous property of a connected network. "
 "Therefore, a research gap remains in representing hydrological connectivity as "
 "a continuous relational structure within resilience models of the belt, since "
 "this study establishes that connectivity status materially conditions "
 "ecological outcome."),

("10.3390/land14030598",
 "Zhu et al. [{n}] (2025) analysed the spatiotemporal dynamics of land surface "
 "temperature and the kernel normalised difference vegetation index across the "
 "Yangtze River Economic Belt through multiple complementary methods. Their "
 "regression analysis established a statistically significant kernel vegetation "
 "index increase of 0.003 per year and a land surface temperature rise of 0.065 "
 "degrees Celsius per year, with principal component analysis explaining 74.5 "
 "per cent of variance and identifying vegetation cover and urbanisation as "
 "dominant influences, providing quantitative belt-wide rates against which "
 "independently derived indicator layers can be validated. However, the study "
 "characterises the behaviour of two individual biophysical variables rather "
 "than constructing any composite ecological or resilience index, and its "
 "analytical apparatus is descriptive and linear. Therefore, a research gap "
 "remains in advancing from the trend characterisation of individual "
 "biophysical variables to the estimation of an integrated resilience construct, "
 "while adopting the kernel vegetation index formulation this study validates "
 "for the belt in preference to the conventional index it improves upon."),

("10.3390/su17157114",
 "Zhang and Wu [{n}] (2025) examined the spatiotemporal evolution and "
 "influencing factors of urban ecological resilience in the Yellow River Basin, "
 "which serves as the conventional large-basin comparator to the Yangtze in the "
 "Chinese ecological literature. Their analysis established resilience "
 "trajectories and driver relationships for a basin subject to comparable "
 "development pressure under markedly different hydrological and climatic "
 "conditions, providing the contrast case against which Yangtze findings are "
 "customarily positioned. However, the study relies on statistical indicator "
 "systems at city level without remote sensing measurement, and its "
 "cross-sectional driver analysis does not accommodate nonlinear or "
 "spatially varying effects. Therefore, a research gap remains in conducting "
 "methodologically equivalent, remote-sensing-based resilience assessment across "
 "both major basins, so that inter-basin comparison is not confounded by "
 "differences in measurement approach between the studies being compared."),

("10.1038/s41598-025-05298-1",
 "Kong et al. [{n}] (2025) analysed the heterogeneous effects of new quality "
 "productive forces on ecological resilience in the Yangtze River Delta "
 "Economic Belt using quantile-based estimation. They found that these "
 "productive forces significantly enhance ecological resilience with the "
 "strongest effect at the low-resilience quantile and progressively diminishing "
 "influence at higher quantiles, consistent with a diminishing marginal effect, "
 "which implies that the resilience response to a given driver depends on the "
 "prevailing resilience level. However, the study is econometric and delta-"
 "specific, employing no remote sensing observation, and the quantile-dependent "
 "response is estimated for a single composite driver rather than "
 "characterised across the driver set. Therefore, a research gap remains in "
 "accommodating level-dependent, heteroscedastic driver response within "
 "resilience models, since a conditional mean estimator fitted across the full "
 "resilience distribution will misrepresent effects that vary systematically "
 "with baseline resilience."),

("10.3390/su17198528",
 "Yang [{n}] (2025) examined the spatiotemporal evolution and driving "
 "mechanisms of coupling coordination between green innovation efficiency and "
 "urban ecological resilience in the Yangtze River Delta. The analysis reported "
 "green innovation efficiency rising from 0.252 to 0.692 while urban ecological "
 "resilience rose more modestly from 0.228 to 0.395, with their coupling "
 "coordination degree advancing from mild discordance to primary coordination, "
 "supplying reference magnitudes for delta resilience levels. However, "
 "resilience is derived from statistical indicators through entropy-style "
 "weighting without remote sensing input, and coupling coordination is a "
 "descriptive composite rather than a predictive construct. Therefore, a "
 "research gap remains in grounding delta-scale resilience magnitudes of this "
 "kind in observed biophysical measurement, so that reported resilience levels "
 "are comparable across studies rather than contingent on each study's chosen "
 "indicator weighting."),

("10.3390/land14091769",
 "He et al. [{n}] (2025) analysed spatial and functional heterogeneity in "
 "regional resilience across the Chengdu-Chongqing economic mega region using a "
 "geographic information system based approach. Their assessment found that only "
 "about 1.29 per cent of the region exhibits high resilience, concentrated in "
 "integrated urban-ecological zones such as Chengdu, which supplies scarce "
 "quantitative evidence for the upstream extremity of the Yangtze corridor where "
 "belt-wide studies are typically least well resolved. However, the analysis "
 "addresses composite regional resilience rather than ecological resilience "
 "specifically, and its geographic information system overlay methodology "
 "assigns rather than learns the relative influence of contributing layers. "
 "Therefore, a research gap remains in resolving ecological resilience "
 "specifically within the upstream belt using learned rather than assigned "
 "layer weighting, particularly given that upstream conditions differ "
 "systematically from the downstream contexts that dominate the literature."),

("10.1038/s41598-025-20425-8",
 "Yang et al. [{n}] (2025) introduced a karst-specific remote sensing "
 "ecological index for monitoring ecological quality in southwest China, "
 "motivated by the observation that karst regions are ecologically fragile and "
 "highly sensitive to both natural and anthropogenic disturbance and therefore "
 "poorly served by general-purpose indices. Their tailored formulation improved "
 "the quantitative characterisation of ecological quality and its driving forces "
 "relative to the conventional index in karst terrain, establishing that "
 "index specification must respond to lithological and hydrological context. "
 "However, the index is developed for karst landscapes in isolation and is not "
 "reconciled with indices applied to adjacent non-karst terrain, so no mechanism "
 "is provided for consistent assessment across a region containing both. "
 "Therefore, a research gap remains in constructing regionally adaptive "
 "ecological indices that remain internally comparable across heterogeneous "
 "lithology, which is directly consequential for the Yangtze River Economic "
 "Belt because its upstream reaches are extensively karstic while its "
 "downstream reaches are alluvial, yet belt-wide studies apply a single uniform "
 "index throughout."),

("10.3390/su172210267",
 "Liu et al. [{n}] (2025) studied changes in the remote sensing ecological "
 "index using a combined remote sensing and Markov-FLUS modelling approach, "
 "demonstrating that integrating scenario-based land use simulation with "
 "ecological index modelling provides an effective basis for ecological "
 "conservation and sustainable urban planning in a region undergoing rapid "
 "economic transformation. The study consolidates the now-standard workflow in "
 "which simulated future land use states are propagated into projected "
 "ecological index values. However, the propagation is deterministic and "
 "sequential, with land use simulated first and the ecological index computed "
 "afterwards, so uncertainty in the simulated land use is not carried through "
 "into the ecological projection and no feedback from ecological state to land "
 "use transition is admitted. Therefore, a research gap remains in jointly "
 "modelling land use dynamics and ecological outcome with propagated "
 "uncertainty, rather than chaining a deterministic simulator to a downstream "
 "index calculation."),

("10.3389/fbuil.2025.1690346",
 "Wan et al. [{n}] (2025) constructed a comprehensive evaluation index system "
 "for the development level of urban ecological resilience using panel data from "
 "277 Chinese prefecture-level cities over 2009 to 2023, organised across the "
 "three dimensions of resistance, resilience and adaptability, and analysed "
 "spatial gaps through two-dimensional decomposition. The breadth of the sample "
 "makes this among the largest-sample resilience indicator systems available and "
 "consolidates the three-dimensional decomposition as the field's dominant "
 "conceptualisation. However, the indicator system is populated entirely from "
 "statistical yearbook variables at city level, so ecological resilience is "
 "measured through administrative and economic proxies rather than observed "
 "ecosystem condition, and no remote sensing evidence enters the assessment. "
 "Therefore, a research gap remains in populating the widely adopted "
 "resistance-resilience-adaptability decomposition with directly observed "
 "biophysical indicators, so that the construct is measured through the "
 "ecosystem properties it purports to describe."),

("10.3390/su18136410",
 "Ma et al. [{n}] (2026) investigated how improvements in ecological "
 "compensation efficiency affect urban economic resilience in the Yangtze River "
 "Economic Belt, examining direct effects, transmission mechanisms and spatial "
 "spillovers. Their findings indicated that improving ecological compensation "
 "efficiency enhances both local and neighbouring economic resilience, "
 "documenting the policy instrument through which ecological governance is "
 "financed across the belt and confirming that its effects cross administrative "
 "boundaries. However, the outcome variable is economic rather than ecological "
 "resilience, and the analysis is econometric without spatial ecological "
 "measurement, so the ecological consequence of the compensation mechanism is "
 "not itself assessed. Therefore, a research gap remains in evaluating whether "
 "ecological compensation transfers produce measurable improvement in observed "
 "ecological resilience, which requires spatially explicit resilience "
 "measurement of the kind econometric policy evaluation does not supply."),
]

SYNTH = [
 "Across these forty-two studies, four converging but incomplete strands of "
 "research emerge. The first strand, comprising Li et al. (2026a), Fu et al. "
 "(2026a), Yao et al. (2025), Chen et al. (2026), Li et al. (2026b), Tang et "
 "al. (2025), Peng et al. (2026) and Ma et al. (2026), establishes ecological "
 "resilience as a tractable and policy-relevant construct for the Yangtze River "
 "Economic Belt, and supplies the consensus empirical facts of the region: "
 "resilience is persistently low in absolute terms, follows a shallow "
 "decline-then-rebound or slow-rise trajectory, and exhibits a pronounced "
 "east-over-west gradient. However, this strand operates through statistical "
 "indicator systems, econometric panel estimation and spatially varying "
 "regression, and not one of its members employs deep learning, so nonlinear "
 "indicator interaction and temporal dependence are imposed by the aggregation "
 "design rather than estimated from data.",

 "The second strand, spanning Gong et al. (2025), Xie et al. (2025), Yang et "
 "al. (2026), Wei et al. (2026), Ke et al. (2026), Soula et al. (2026), Yuan et "
 "al. (2026), Kaur and Sharma (2025), Aldossary (2025), Li et al. (2026c) and "
 "Yang (2026), demonstrates that deep architectures are mature for ecological "
 "and geospatial estimation, encompassing learned nonlinear index construction, "
 "theory-constrained deep aggregation, graph attention over geographic units, "
 "bidirectional recurrent-convolutional multi-scale prediction, "
 "domain-constrained loss design and foundation-model embeddings. However, every "
 "member of this strand targets ecological quality, land cover, land use "
 "transition or a single vegetation index, and none estimates ecological "
 "resilience; the one study conducted at full belt extent with a deep "
 "architecture, Wei et al. (2026), predicts land use rather than resilience and "
 "reports an aggregate accuracy metric insensitive to the changed pixels that "
 "carry the ecological signal.",

 "The third strand, comprising Qu et al. (2026), Cheng et al. (2026), Tong et "
 "al. (2025), Fu et al. (2026b), Li et al. (2026d), Ding et al. (2026), Yue et "
 "al. (2026) and Kong et al. (2025), provides convergent diagnostic evidence "
 "that ecological change in the region is fundamentally nonlinear: long-run "
 "index means remain near-stationary while fluctuating markedly, drivers act "
 "through identifiable thresholds, causal structure among drivers evolves over "
 "time, and driver effects vary systematically with baseline resilience level. "
 "This strand identifies the phenomenon but addresses it with post hoc "
 "attribution over tree ensembles, so nonlinearity is characterised after the "
 "fact rather than represented within a model whose inductive bias matches it, "
 "and the spatial autocorrelation that Fu et al. (2026b) explicitly confirm in "
 "the resilience field is diagnosed and then discarded by estimators that treat "
 "spatial units as independent.",

 "The fourth strand, encompassing Wang et al. (2026a), Yang et al. (2025), Zhu "
 "et al. (2025), Guo et al. (2025), Yu et al. (2026), Li et al. (2025), Liu et "
 "al. (2025), Wan et al. (2025), He et al. (2025), Zhang and Wu (2025), Zhao et "
 "al. (2026), Wang et al. (2026b) and Yang (2025), supplies the measurement and "
 "simulation infrastructure: validated belt-wide biophysical rates, adaptive "
 "indicator reduction, region-specific index formulations for karst terrain, "
 "hydrological connectivity effects, coupled risk-resilience indices and the "
 "conventional Markov-FLUS simulation baseline. However, this strand computes "
 "its composites through entropy weighting, analytic hierarchy process, "
 "principal component analysis or TOPSIS, so indicator weights are "
 "analyst-specified; only Yue et al. (2026) tests the robustness of conclusions "
 "to weighting choice, and only Wang et al. (2026a) reduces indicators "
 "objectively, and in both cases the procedure remains decoupled from any "
 "downstream predictive model.",

 "Taken together, the literature establishes every component required for "
 "deep-learning-based ecological resilience assessment of the Yangtze River "
 "Economic Belt without integrating them. Belt-scale resilience assessment "
 "exists but is statistical; deep spatiotemporal architectures exist and have "
 "been shown to train at belt extent but are pointed at land use rather than "
 "resilience; the nonlinearity that would justify deep modelling is documented "
 "but only diagnosed; and the measurement infrastructure is available but "
 "aggregated through subjective weighting. Assessed against the four axes of "
 "study area, resilience target, remote sensing measurement and deep learning "
 "method, no study located in this review satisfies more than three "
 "simultaneously. Four further gaps compound this structural absence. First, "
 "spatial autocorrelation in the resilience field is confirmed yet never "
 "modelled, and no study constructs a relational graph over administrative "
 "units despite Ke et al. (2026) and Kaur and Sharma (2025) demonstrating the "
 "means. Second, indicator weighting remains subjective throughout. Third, "
 "internal heterogeneity is ignored, since Yang et al. (2025) establish that "
 "karst terrain requires its own index formulation while the extensively karstic "
 "upstream of the belt is nonetheless assessed with the same uniform index as "
 "its alluvial downstream. Fourth, terrestrial and aquatic resilience are "
 "modelled in separate literatures and their gradients disagree, with Li et al. "
 "(2026a) reporting an east-over-west terrestrial ordering while Tang et al. "
 "(2025) report a downstream-upstream-midstream ordering for water resilience in "
 "the same study area. Therefore, a research gap remains in developing and "
 "rigorously evaluating a spatiotemporal deep learning framework that estimates "
 "county-level ecological resilience across the entire Yangtze River Economic "
 "Belt from multi-source remote sensing observation, that learns rather than "
 "prescribes the aggregation of resistance, recovery and adaptability "
 "components, that represents spatial and hydrological connectivity explicitly "
 "rather than assuming unit independence, that adapts to the belt's "
 "lithological and ecological heterogeneity rather than imposing a uniform "
 "index, and that is benchmarked against the statistical, interpretable-machine-"
 "learning and simulation baselines this literature has established rather than "
 "evaluated in isolation.",
]


def main():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    st.paragraph_format.space_after = Pt(10)
    st.paragraph_format.line_spacing = 1.5

    h = doc.add_heading(TITLE, level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for doi, body in P:
        n = NUM[doi]
        para = doc.add_paragraph(body.replace("{n}", str(n)))
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    doc.add_heading("Synthesis of the literature", level=2)
    for s in SYNTH:
        para = doc.add_paragraph(s)
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    doc.add_heading("References", level=2)
    for doi in ORDER:
        rec = H[doi]
        para = doc.add_paragraph(f"[{NUM[doi]}] {rec['harvard']}")
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_after = Pt(6)

    out = "Literature_Survey_Ecological_Resilience_YREB.docx"
    doc.save(out)

    words = sum(len(b.split()) for _, b in P) + sum(len(s.split()) for s in SYNTH)
    print(f"saved {out}")
    print(f"paper paragraphs: {len(P)} | references: {len(ORDER)} | "
          f"body words: ~{words}")
    missing = set(NUM) - {d for d, _ in P}
    print(f"papers discussed but unnumbered: {missing or 'none'}")
    assert len(P) == len(ORDER) == 42, "count mismatch"
    assert {d for d, _ in P} == set(ORDER), "DOI set mismatch"
    print("consistency checks passed")


if __name__ == "__main__":
    main()
