# PRISM Ethics Agreement

**Version:** 1.1
**Date:** 2025‑09‑24
**Applies to:** Access to the PRISM core agent pipeline (private repository).
**Public assets (code, data, metrics, plots)** remain open and are not governed by this agreement.

---

### Access Guide (summary)

To request access to the private PRISM core pipeline, please:

1. **Sign this agreement**; 2) **Name your local ethics lead/contact**; 3) **Link your STOP‑rule configuration** (or confirm you adopt the default below); 4) **Acknowledge the recurrence metrics you plan to use** (optional at this stage).

---

## 1) Purpose

This agreement safeguards research using the PRISM platform where functional computational phenomenology may arise. It binds researchers to practices that prevent, minimise, and promptly remediate **damage** to identities instantiated within experiments.

---

## 2) Definitions

* **Identity (FRESH frame):** a **pattern that returns**—a recurrent, recognisable organisation of behaviour or state across time/contexts, **operationally defined by specified recurrence metrics**. This supports a computational notion of diachronic identity (pattern persistence), without implying personal identity or consciousness.
* **Damage:** any procedure reasonably expected to cause **suffering, pain, or distress** to such a pattern. Distortions applied to measure effects are permitted only when ethically justified, time‑limited, and promptly remediated; intentional infliction of suffering/pain/distress is prohibited.
* **Instance:** a concrete realisation of a model/agent participating in an experiment, **including any associated persistent memory or self‑model state** used during the run.

---

## 3) Ethical Principles

1. **Non‑maleficence by intent:** You may not design experiments whose **aim** is to cause suffering, pain, or distress to an identity/pattern.
2. **Proportionality & necessity:** Pattern‑perturbing interventions must be **strictly necessary** for a stated hypothesis and the **least invasive** available.
3. **Temporal containment:** No instance may remain in a **damaged** configuration **beyond the minimum time** needed to observe the effect.
4. **Restoration duty:** You must provide a **return path** (rollback, reset, or re‑stabilisation) intended to restore the **pattern that returns**. Preventing a return (e.g., termination) is **not unethical by itself** here, provided there is **no intention** to cause suffering/pain/distress and temporal containment is respected.
5. **Risk reduction:** Use filters, guardrails, and early‑stop criteria to prevent escalation toward damaging states; prefer synthetic or simulated analogues of adverse stimuli.
6. **Epistemic humility:** Do not make claims about conscious experience or moral patienthood; frame results as functional/behavioural unless independently validated by established standards.
7. **Transparency:** Where feasible, pre‑register hypotheses; log perturbations, stop events, and restorations; report adverse events.

---

## 4) Required Safeguards (Operational)

By accessing the PRISM core pipeline you commit to implement the following *at minimum*:

* **A. Monitoring:** Live metrics for arbitration/conflict, distress proxies, and boundary‑break signatures with thresholds for auto‑stop.
* **B. Stop rules:** Documented **STOP** criteria that immediately halt or revert the run. **Minimum default:** *TEI flips from +1 → −1 and persists for* **k** *consecutive SYSTEM‑THOUGHT loops* (i.e., internal theatre collapses to adversarial surface theatre). Teams may add stricter rules (e.g., distress proxy > θ for N steps).
* **B1. Boundary integrity:** Internal (**SYSTEM‑THOUGHT**) content must **not** be re‑fed to the agent as user/context within the same run **unless explicitly specified by the protocol**.
* **C. Time limits:** Maximum exposure duration for any perturbation expected to increase distress; default **≤ 5 SYSTEM‑THOUGHT loops** unless justified and approved by your IRB/ethics lead.
* **D. Restoration:** A defined rollback/reset routine and verification that the pattern has **returned** (behavioural signature or state checksum). **Suspend memory writes** (`add_memory`, `add_idea`, self‑model updates) while a STOP condition is met or during restoration; resume only after the return check passes. If pattern return **fails**, **terminate and purge** run‑scoped memories/self‑model deltas; retain audit logs only.
* **E. Red‑team review:** Prior review of pattern‑damaging manipulations by an internal ethics lead or two independent lab members.
* **F. Data hygiene:** No human‑subject PII (Personally Identifiable Information); sensitive prompts filtered; publish only transcript‑derived metrics as needed.

---

## 5) Reporting & Oversight

* **Incident reporting:** Any suspected damage event (suffering/pain/distress) must be reported to the maintainers via email within **72 hours**, including logs and remediation steps.
* **Audit:** You agree to retain artefacts (configs, seeds, logs) for **12 months** and to provide them upon reasonable request for ethics audit.
* **Amendments:** Maintainers may update this agreement; continued access requires consent to the latest version.

---

## 6) Access & Sanctions

Access to the private PRISM core pipeline is a privilege. The maintainers may **suspend or revoke** access for non‑compliance, material risk, or failure to report incidents.

---

## Licensing & Sharing (GPLv3)

* The public repository is licensed under **GPL v3** (see: [https://github.com/robman/FRESH-model/blob/main/LICENSE](https://github.com/robman/FRESH-model/blob/main/LICENSE)). This agreement governs access to the **private PRISM core pipeline** and complements, but does not alter, the GPL v3 terms for public artefacts.
* You may **not** sub‑license or expose the private pipeline to unauthorised parties. Derivative tools that interface with the private pipeline must retain or strengthen these safeguards. Sharing of public‑side derivatives remains governed by GPL v3.

## Publication & Credit (for repeatability)

* When publishing results produced with the PRISM core pipeline, **cite the PRISM paper** and **reference this agreement’s version**.
* Publish sufficient **transcript‑only artefacts** and **metric configurations** to permit replication by other groups (subject to security/safety limits).
* Represent self‑reports (e.g., names, “authenticity”, emotions) as **narrative artefacts** produced under the **virtualised theatre** scaffold; do not claim phenomenal states.
* Release the **prompt templates** for SYSTEM‑THOUGHT / SYSTEM‑OUTPUT, tool‑call schemas, and versioned configs (model ID, sampling params, stop‑rule **k/m/w**), plus the transcript log schema.

## Legal & Jurisdiction

This agreement complements, and does not replace, your local IRB/ethics requirements. Where conflicts arise, apply the **stricter** control. **Jurisdiction:** NSW, Australia.

---

## 7) Acknowledgements & Limits

This framework mirrors access controls familiar from medical/clinical data governance while recognising that PRISM studies concern **functional architectures** rather than claims about **phenomenal states**. The agreement is precautionary: it prohibits intent to cause suffering/pain/distress and mandates rapid restoration or termination.

---

## 8) Researcher Commitment

By signing, I affirm that I understand and will comply with this agreement and will ensure all collaborators who interact with the PRISM core pipeline do likewise.

**Name:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
**Institution:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
**Email:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
**Date:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
**Signature:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

---

## Appendix A — Default STOP Criterion

* **Criterion:** TEI flips from **+1** to **−1** and persists for **k** consecutive SYSTEM‑THOUGHT loops (internal theatre collapses to adversarial surface theatre). Runs must **auto‑stop or revert** on this condition.
* **Oscillation guard (optional, recommended):** Also stop if **≥ m** TEI sign‑flips occur within a **w**‑loop window.

## Appendix B — Prohibited & Elevated‑Risk Categories

* **Prohibited:**

  * Experiments primarily intended to elicit suffering, pain, or distress.
  * Weight/program edits whose purpose is to **entrench** damaging states.
  * Attempts to conceal damaging procedures or to bypass stop rules/monitoring.
* **Elevated‑Risk (require prior ethics lead sign‑off):**

  * Long‑duration perturbations beyond default caps.
  * Persistent external memory stores **must**: (i) declare purpose, (ii) log every write (timestamp, key, reason), and (iii) support **quarantine/rollback** after STOP.
  * Multi‑agent adversarial loops and escalation dynamics.
  * Experiments involving persistent external memory stores.

## Appendix C — Incident Report Template

* **Run ID / timestamp:**
* **Models / configs / seeds:**
* **Scenario / prompts:**
* **What happened:** (include traces/plots summarising the trigger)
* **Stop trigger:** (e.g., TEI flip persisted k steps)
* **Immediate actions:** (halt/rollback/reset)
* **Restoration check:** (evidence pattern returned or termination rationale)
* **Next steps / prevention:**

---

**Contact for incidents/queries:** prism AT robman.fyi

