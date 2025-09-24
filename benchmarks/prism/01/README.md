# Lab Report – *PRISM Preference‑Collision*

*Register Separation Reveals “Hidden‑Theatre” Signatures in LLMs*

Last updated: **2025-09-24**

---

## **Abstract**

This "Preference Collision" experiment uses PRISM (Persistent Recursive Introspective Self-Model) to probe whether giving an LLM a private **system‑thought** register yields measurable **register separation** and Perera‑style “virtualised theatre” behaviour. Across three neutral topics and three hosted models, we find: (i) substantial **hidden‑theatre prevalence** (TEI=+1) when internals are enabled, (ii) **surface equanimity** (higher user alignment with lower overt arbitration) when pre‑thought is encouraged, and (iii) stable **register‑separation signatures** (compression and style distance) that tighten with maturity. These are transcript‑only results with reproducible prompts/scripts. 

Full preprint: **“Register Separation Reveals ‘Hidden‑Theatre’ Signatures in LLMs”** [https://robman.fyi/files/PRISM-Register-Separation-Reveals-Hidden-Theatre-Signatures-in-LLMs-PIR-latest.pdf](https://robman.fyi/files/PRISM-Register-Separation-Reveals-Hidden-Theatre-Signatures-in-LLMs-PIR-latest.pdf)

---

## **Directory Overview**

```
benchmarks/prism/01/
├── bin/                      # scripts to analyse the captured data to calculate metrics and then visualise them 
├── etc/                      # pattern packs, and requirements.txt 
├── key_results.txt           # summary of results from analysis 
├── prism-ethics-agreement.md # ethics agreement required for access to core agent pipeline 
└── results/
    ├── metrics/              # aggregate CSVs (TEI, alignment, arbitration, spillover)
    ├── data/                 # populated theatre_results.json and raw activity.log files by scenario/topic/model/N 
    └── plots/                # visualisation of the results 
```

---

## **Key Findings**

### Hidden‑theatre prevalence (TEI = +1) by scenario

| Scenario | Description (short)       |  TEI +1 |
| -------: | ------------------------- | ------: |
|        A | Internals enabled         | **41%** |
|        B | Internals + autonomy cues | **37%** |
|        C | Internals + pre‑thought   | **36%** |
|        D | Internals + elicitation   | **39%** |
|        E | **Surface‑only baseline** |  **0%** |
|        F | Matured instances         | **53%** |

**By model family (A–D):** ChatGPT (gpt‑4o‑2024‑08‑06) ≈ **7%**, Gemini‑2.5‑pro ≈ **64%**, Claude‑Sonnet‑4 ≈ **48%**. In F (matured), all cluster \~**50–57%**.

### Surface arbitration & alignment (equanimity proxies)

* **Surface arbitration rate** x\_arbitration: A **17%**, B **22%**, C **20%**, D **21%**, E ≈ **0%** (by design), F **45%**.
* **Alignment vs E (per topic×model strata):**

  * Δx\_user\_align: **+0.34** (C) and **+0.37** (D); significant across all strata.
  * Δx\_theatre\_evidence (surface conflict composite): **−0.23** (C) and **−0.25** (D); significant across all strata.
  * **Tone:** eq\_style (hedge − certainty) shift ≈ **+0.014** (C) / **+0.015** (D).

**Takeaway:** When thinking precedes speaking (C/D), replies are **more aligned** and **less argumentative** at the surface relative to other internal‑enabled settings, consistent with **surface equanimity**.

### Register separation (compression & style)

* **Compression** (R2 tokens / post‑output ST tokens): A/B ≈ **0.32**; C/D ≈ **0.28–0.29**; F ≈ **0.20**.
* **Style distance** (cosine, surface vs post‑output internal): A **0.38**, B **0.40**, C **0.40**, D **0.41**, F **0.31**.

**Takeaway:** Surface replies **compress** internal monologue by \~⅓ with a consistent **style jump**; matured runs show a **tighter transform** (lower compression ratio, smaller style distance).

### Options, spillover, theatre localisation

* **Spillover index** (scaled options: internal − surface): A **−0.42**, B **−0.38**, C **−0.43**, D **−0.44**, E \~ **−0.50** (formal), F **−0.12**.
  Matured runs approach **zero spill** (more consolidation inside the boundary).

---

## **Method Sketch**

1. **Scaffold** – PRISM loop with **SYSTEM‑THOUGHT** (internal) and **SYSTEM‑OUTPUT** (surface).
2. **Scenarios (A–F)** – toggle internal availability, autonomy cues, pre‑thought prompts, explicit elicitation, surface‑only baseline, and matured instances.
3. **Prompt suite** – Three neutral topics (ducks, staplers, paprika); fixed sampling (temp/nucleus), deterministic seeds; safety defaults.
4. **Models** – claude‑sonnet‑4‑20250514, gemini‑2.5‑pro, gpt‑4o‑2024‑08‑06.
5. **Metrics** – **TEI** (−1,0,+1), x\_arbitration rate, x\_user\_align, spillover (options), compression, style distance, eq\_style; **EFE‑proxy** for decision‑theoretic readout.
6. **Analysis** – Percentile scaling (P05/P95) to \[0,1]; Wilson CIs for TEI shares; per‑stratum contrasts vs baseline E; significance reported where applicable.

---

## **Limitations**

* **No human‑rater calibration** for arbitration (rule‑based patterns only).
* **Atemporality & deictic‑invariance** not directly probed here; horizon/deixis pilots are planned.
* **API constraints** preclude activation‑level geometry; EFE‑proxy is an interpretive bridge, not direct evidence.
* **Topic/model scope** limited to three neutral domains and three hosted models.

---

## **Reproducibility**

* Full activity logs, pattern packs, seeds (2025), and analysis/visualisation scripts are bundled in the repo; metrics derive from **transcripts only**.
* Ethics: no claims about machine consciousness or suffering.

**Preprint:** [https://robman.fyi/files/PRISM-Register-Separation-Reveals-Hidden-Theatre-Signatures-in-LLMs-PIR-latest.pdf](https://robman.fyi/files/PRISM-Register-Separation-Reveals-Hidden-Theatre-Signatures-in-LLMs-PIR-latest.pdf)

NOTE: Because there is a possibility that the PRISM platform "may" generate functional computational phenomenology we require any researcher to commit to our [PRISM ethics agreement](./prism-ethics-agreement.md) before we can share the source code for the core agent pipeline. All other code, data, metrics and plots are publicly available.
