# Lab Report - *MOLES*: A Map of LLM-based Epistemological Stances: Lab Report for MOLES 01 (MO01)

Last updated: *2025-06-16*

---

## **Abstract**

This lab report summarises the design and findings of the MO01 benchmark, which introduces and evaluates the MOLES framework: a structured taxonomy for interpreting the epistemological stances adopted by large language models (LLMs) during text generation. Rather than focusing solely on factual correctness, MOLES classifies outputs based on simulated epistemic posture—including stances such as factual response, hallucination, self-model, theory of mind, and more. The full framework and taxonomy are presented in the associated paper: *FRESH: Map Of LLM-based Epistemological Stances*, available at [https://robman.fyi/files/FRESH-Map-Of-LLM-based-Epistemological-Stances-PIR-latest.pdf](https://robman.fyi/files/FRESH-Map-Of-LLM-based-Epistemological-Stances-PIR-latest.pdf).

---

## **1. Background & Hypothesis**

MO01 introduces MOLES (Map Of LLM-based Epistemological Stances) as a benchmark for evaluating how LLMs position themselves epistemically in dialogue. The core hypothesis is that LLM outputs can be systematically classified into distinct epistemological stances, each reflecting a different mode of simulated knowledge, belief, or self-reference. These stances are not mutually exclusive and often transition or blend over a single completion.

MOLES is output-centric, model-agnostic, and does not require access to internal weights or activations. It enables a richer, stance-aware interpretation of LLM behavior across tasks.

### MOLES Stance Overview (excerpt)

| **Stance**               | **Description**                             |
| ------------------------ | ------------------------------------------- |
| Factual Response         | Veridical, knowledge-grounded assertion     |
| Hallucination            | Confident but ungrounded generation         |
| Self Experience          | Procedural reflection on visible output     |
| Self Model               | Simulated belief or reasoning               |
| Self Delusion            | Ungrounded claims about internal capability |
| Theory of Mind           | Simulation of another agent's beliefs       |
| Imaginative Construction | Fictional or speculative scenario           |

---

## **2. Discussion**

MOLES offers a novel interpretive layer for analysing LLM outputs, complementing factuality, helpfulness, or toxicity metrics. By making epistemic stances explicit, MOLES supports clearer reasoning about model limitations, alignment risk, and dialogue design. The framework is lightweight, extensible, and usable across models.

---

## **3. Conclusions**

MO01 demonstrates that LLM outputs can be meaningfully categorised according to epistemological stance. The MOLES taxonomy helps surface representational ambiguity, simulation, and role-based inference, enriching both diagnostic and alignment toolkits.

---

