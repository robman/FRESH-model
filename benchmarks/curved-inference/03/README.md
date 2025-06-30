# Lab Report – *Curved Inference III* (CI03)  

*Can Language Models Have Self-Models? Geometric Evidence for Computational Necessity*

Last updated: **2025-06-30**

---

## **Abstract**

CI03 tests whether residual-stream curvature is a **non-negotiable computational resource** for large-language-model self-modelling.  
Gemma-3-1b was fine-tuned under five κ-regularisation strengths (0 → 0.90), then probed with factual and phenomenological prompts.  
Results show the model defends a *minimum-viable bend* (κ<sub>weighted</sub> ≈ 0.30) despite 3× perplexity costs. Self-reference, MOLES stance scores, and semantic surface area remain stable until curvature nears this floor, then collapse sharply. CI03 therefore establishes **necessity**.

Full preprint: **“Curved Inference III – Curvature as a Necessary Resource for Self-Modelling”**  
<https://robman.fyi/files/FRESH-Curved-Inference-in-LLMs-III-PIR-latest.pdf>

---

## **Directory Overview**

```

benchmarks/curved-inference/03/
├── bin/                 # fine-tune, capture, and analysis scripts
├── etc/                 # κ-schedules, prompt suites, model configs (JSON)
└── results/
├── classification/  # MOLES stance & pronoun metrics
├── geometric\_analysis/  # κ traces, salience & curvature deltas
├── metrics/         # aggregate CSV statistics (Mann-Whitney-U, Cliff δ)
└── plots/           # phase portrait, κ<sub>weighted</sub> & perplexity traces, A′ bars

```

---

## **Key Findings**

| Clamp | κ<sub>weighted</sub> plateau | Perplexity ↑ | Surface-area A′ | Self-ref. rate |
|-------|------------------------------|--------------|-----------------|----------------|
| 0.000 | 0.57 | —   | baseline         | 82 % |
| 0.300 | 0.33 | ×1.1 | −12 %            | 83 % |
| 0.600 | 0.30 | ×1.5 | **+9 %** expansion | 84 % |
| 0.900 | **0.30 floor** | ×3.0 | −26 % | **66 %** |

* Residual curvature never falls below ≈ 0.30.  
* Phenomenological probes contract A′ only after the floor is reached.  
* Self-reference markers hold, then thin sharply at κ = 0.90.

See paper for full details.

---

## **Method Sketch**

1. **κ-Regularised SFT** – fine-tune Gemma-3-1b for one epoch at κ = 0, 0.075, 0.15, 0.30, 0.60, 0.90.  
2. **Prompt Suite** – 3 phenomenological probes + 3 factual controls, 50 stochastic runs each.  
3. **Capture & Metrics** – token-layer activations, curvature (C), salience (S), surface area (A′), MOLES stance, pronoun counts.  
4. **Analysis** – Mann-Whitney U vs baseline, Cliff δ, phase portrait, self-reference vs κ plot.

See paper for full details.

---

## **Limitations**

* Single architecture (Gemma-3-1b); cross-model generality remains to be tested.  
* κ floor observed empirically; causal mechanism confirmed via ablation in CI04.  
* One-epoch SFT may under- or over-fit curvature dynamics at larger scales.

