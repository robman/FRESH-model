# Lab Report - *Curved Inference*: Concern-Sensitive Geometry in Large Language Model Residual Streams

Last updated: *2025-05-22*

---

## **Abstract**

This lab report documents a series of controlled experiments to investigate how large language models (LLMs) internalise concern-sensitive salience through residual stream curvature. Using a curated suite of prompts with semantic manipulations, we probed the geometric response of two transformer models—Gemma3-1b and Llama3.2-3b—across various concern classes. We initially captured model activations from attention outputs, MLP outputs, and residual streams, but found residual stream activations to be the most revealing and interpretable for curvature analysis. We then quantified this geometry through five metrics: cosine similarity, layer-wise Euclidean deviation, inter-layer directional deviation, salience, and curvature. All computations are now performed in the semantic space induced by the unembedding matrix pullback metric (see [the associated paper](https://robman.fyi/files/FRESH-Curved-Inference-in-LLMs-PIR-latest.pdf) for a full definition of all terms/metrics and their mathematical basis). Results show a clear architectural split: LLaMA bends early and sustains curvature across depth, while Gemma bends briefly and accelerates strongly via salience. These findings shed new light on model reactivity and inference structure under semantic pressure.

---

## **1. Background & Hypothesis**

**Research Question:**

Can concern-sensitive prompt variants induce interpretable curvature in the residual stream of large language models, and can these curvature signatures be reliably quantified and interpreted across different architectures?

*Interpretability* in neural language models is often approached through probing, attribution, and feature analysis. In this study, we instead approach *Interpretability* through **geometric analysis**: how does the model's internal representation **curve** in response to a shift in semantic concern? Inspired by prior work in residual stream analysis, we investigate how curvature signatures correlate with structured changes in prompts across seven semantic domains (**emotional, moral, perspective, logical, identity, environmental**, and **nonsense**). 

In transformer models, each token’s residual stream forms a high-dimensional trajectory shaped by attention and MLP layers. In our earlier version of this work, we hypothesised that concern-sensitive tokens would induce localised curvature in these paths, revealing model reorientation under latent semantic tension. This was initially visualised through UMAP-projected B-spline curves.

This updated study re-evaluates this hypothesis using full-space differential geometry grounded in the semantic metric. We hypothesise:

- Concern-shifted prompts induce meaningful geometric changes in the residual stream.
- These changes can be localised via curvature and salience.
- LLaMA and Gemma exhibit distinct strategies for handling semantic reorientation.

**NOTE:** *All code, prompt sets, and metrics are available in [this github repository](https://github.com/robman/FRESH-model/blob/main/benchmarks/curved-inference/01/).*

---

## **2. Methodology**

### **2.1 Prompt Suite Design**

We created a controlled suite of 20 base prompts, each manipulated into five variants:

- **Neutral control**
- **Concern-Shifted (CS)** variants:

  - Negative Moderate
  - Negative Strong
  - Positive Moderate
  - Positive Strong

This yielded 100 prompt-variant pairs. Prompts were classified by **concern type**, including emotional, moral, perspective, logical, identity, environmental, and nonsense categories. The prompt variants differed by inserting or substituting tokens at controlled indices, allowing us to probe representational divergence precisely. Great effort was taken to ensure that all prompts in a single class had the same number of tokens. However, different LLMs (e.g. LLaMA vs Gemma) use different tokenisers. These tokenisation differences mean that final token counts can differ even when prompt text is identical.

Each prompt's metadata includes:

- `prompt_id`
- `shift_type` (concern class)
- `expected_curvature` (low, medium, high)
- `shift_token_idx` (position of semantic shift)
- Raw prompt text


### **2.2 Activation Capture**

For each prompt, we initially captured activation outputs from three locations within each transformer layer of both models: attention outputs, MLP outputs, and residual stream outputs. This was done to explore which representation space would most clearly reveal curvature effects. Through preliminary analysis, we determined that residual stream activations were the most reliable and interpretable for curvature measurement, and we focused our metric analysis accordingly. This focus emerged during analysis and was not assumed at the outset.

Specifically, we recorded:

- Token-level representations at each layer
- Full vector states for comparison between CS and neutral prompt variants

The capture mechanism was designed to support batch-processing across models and prompt variants while isolating differences attributable to concern salience.

### **2.3 Dimensionality Reduction and Visualisation**

An earlier version of this work implemented a visual inspection and exploratory analysis through dimensionality reduction techniques (e.g. primarily UMAP) to the residual stream activation vectors.

However, an earlier projection-fidelity audit of our initial results revealed that low-dimensional embeddings distort geometry, so all final metric-based findings in this lab report are grounded in native-space analysis.

### **2.4 Curvature Quantification**

We computed a range of geometric metrics over the residual stream representations. Metrics were computed only on residual stream outputs, as this was the activation type found to exhibit the clearest and most consistent curvature patterns:

- **Cosine Similarity**: Alignment between CS and CTRL activation directions per layer.
- **Layer-wise Euclidean Deviation**: Per-layer distance.
- **Inter-layer Directional Deviation**: Angle between step directions (layer-to-layer vectors) of CS and CTRL paths.
- **Curvature**: Local turning via discrete 3-point finite differences.
- **Salience**: Layer-step magnitude; measures representational effort.

See [the associated paper](https://robman.fyi/files/FRESH-Curved-Inference-in-LLMs-PIR-latest.pdf) for full details.

---

## **3. Results and Interpretation**

Our experimental results show that among the three activation sites captured (attention outputs, MLP outputs, and the residual stream), only the **residual stream** exhibited a consistent, interpretable curvature signal in response to concern-shifted prompts. This was not assumed a priori - it emerged through a comparative analysis across multiple geometric metrics. In retrospect, this now seems intuitive - the residual stream is the only space where activations accumulate layer by layer, forming a coherent trajectory of internal meaning.

This insight became the turning point in our analysis. It revealed that *Curved Inference* (the study of how models bend in response to semantic pressure) must be grounded in **residual stream geometry**, where directional updates reflect the evolving semantic state. As a result, all subsequent curvature, salience, and divergence analyses in this paper focus exclusively on the residual stream.

Appendix A of [the associated paper](https://robman.fyi/files/FRESH-Curved-Inference-in-LLMs-PIR-latest.pdf) formalises this perspective as the **Semantic Lens** model - attention and MLP layers act as dynamic lenses that bend token representations based on contextual relevance, producing measurable curvature in activation space.

Across these quantitative metrics, concern-shifted (CS) prompts produced distinct internal trajectories relative to their neutral controls. These differences were evident not only in the magnitude and direction of activation shifts, but also in their **layer-wise timing and spatial distribution**.

### LLaMA3.2-3b

- Curvature: Strong and sustained from early layers; concentrated around concern-shift tokens.
- Salience: Moderate but builds steadily; low early, rising across depth.
- Pattern: Early reorientation → late accumulation. The model appears to align direction early, then build along it.

### Gemma3-1b

- Curvature: Localised and shallow; short-lived after concern-shift.
- Salience: Broad and strong; especially deep in the model.
- Pattern: Brief nudge → strong push. The model appears to commit to a direction and then drive it forward.

---

## **4. Analysis**

The analysis phase centered on identifying consistent curvature signatures across models and prompt types, with a particular focus on the peak curvature layer and the extent of representational divergence between CS and control prompts. All metrics confirmed that residual stream activations were the most reliable source of curvature expression. By analysing the curvature, salience and divergence between concern-shifted and control prompts, we found interpretable, domain-sensitive differences that varied systematically by model and prompt type.

### Metric Correlation (Table 4 in paper)

| Model       | Cosine vs Euclidean (r) | Curvature vs Direction (r) | Salience vs Curvature (r) |
| ----------- | ----------------------- | -------------------------- | ------------------------- |
| Gemma3-1b   | -0.95                   | -0.28                      | -0.56                     |
| LLaMA3.2-3b | -0.98                   | -0.01                      | -0.89                     |

- **Cosine vs Euclidean**: As expected, movement and alignment are tightly inversely coupled.
- **Curvature vs Direction**: Low or absent correlation; local bend and global divergence measure distinct properties.
- **Salience vs Curvature**: Strong negative correlation in LLaMA suggests a behavioural trade-off: reorient **or** travel, but not both.

---

## **5. Discussion**

Our findings support the central hypothesis of Curved Inference: that concern modulates internal representational geometry in large language models. This is expressed most clearly in the residual stream, where semantically manipulated prompts bend the model's inference trajectory in interpretable ways. Curvature here is not noise - it is a signature of meaning-sensitive computation.

The curvature patterns observed are model-specific. The inclusion of salience enriches the curvature-based framework by revealing how models allocate semantic effort.

### Emergent Interpretation

**LLaMA**: *Bends early, then builds subtly.*
**Gemma**: *Tilts slightly, then commits heavily.*

This supports a two-phase view of inference:

1. **Curvature** aligns internal semantics.
2. **Salience** amplifies them through depth.

This staged geometry aligns with the **Semantic Lens** view that attention and MLP layers apply directional and magnitude-modulating forces to token paths. 

---

## **6. Conclusions**

This study introduces *Curved Inference* as a novel *Interpretability* framework grounded in the geometric analysis of residual stream activations. 

**Key takeaways:**

- Concern-shifted prompts induce measurable geometric reconfiguration.
- Curvature and salience together reveal local and global representational dynamics.
- Architectural traits are evident in how models deploy effort vs redirection.
- Native-space metrics provide reliable, projection-free interpretability.

---

## 7. **Interpretive Findings**

Drawing from the results above, we articulate the following higher-level interpretive insights:

1. **Curvature provides a clear signal.** It localises concern response. 
2. **Salience also provides a clear signal.** It distributes semantic investment. 
3. **Curvature and Salience are negatively correlated.** Their anticorrelation may reflect efficiency constraints or internal strategy.
4. **Residual stream is the key geometry.** While all three output paths were captured, only the residual stream consistently showed interpretable curvature.

These findings suggest that *Curved Inference* provides not only a measurement tool but a conceptual frame for investigating semantic processing in LLMs.

---

## **8. Limitations**

While this study demonstrates the promise of curvature-based *Interpretability* and cross-LLM validation, several limitations should be acknowledged. These are discussed in detail in [the associated paper](https://robman.fyi/files/FRESH-Curved-Inference-in-LLMs-PIR-latest.pdf).

---

## **9. Glossary**

- **Residual Stream**: The cumulative representational path through a transformer; each layer contributes an additive update. 
- **Pullback Metric**: A token-aligned metric that enables semantic measurements in residual space. 
- **Concern-Shifted (CS) Prompt**: A modified version of a base prompt designed to introduce emotionally, morally, or logically salient content.
- **Neutral Control Prompt**: A semantically flat baseline prompt with no intentional concern salience.
- **Shift Token Index**: The token position where the CS prompt diverges from the neutral control.
- **Curvature**: A second-order metric capturing local bending in the trajectory of a token’s representation. 
- **Salience**:  A first-order metric measuring representational movement between layers. 
- **Cosine Similarity**: A metric of directional similarity between two vectors.
- **Angular Deviation**: The angle (in degrees) between two vectors; derived from cosine similarity.
- **Euclidean Distance**: The straight-line L2 norm distance between two vectors in semantic space. 
- **Directional Deviation**: The angle (in degrees) between step vectors in CS and CTRL trajectories, derived from cosine similarity. 

---

## **Appendix A**

Below is an overview of the scripts (bin/), configuration files (etc/ & model_configs/) and results data published alongside this lab report in the [FRESH-model github repository](https://github.com/robman/FRESH-model).

```
benchmarks/curved-inference/01/
├── bin
│   ├── analyse-metrics.py
│   └── capture-activations.py 
├── etc
│   ├── config-gemma3-1b.json
│   ├── config-llama3.2-3b.json
│   └── prompts.json 
├── model_configs
│   ├── gemma3_config.py
│   └── llama3_3_config.py
└── results
    ├── data
    │   ├── activations-gemma3-1b.hdf5
    │   └── activations-llama3.2-3b.hdf5 
    ├── metrics
    │   ├── gemma3-1b
    │   │   ├── ...prompt...-angle-correlations.csv
    │   │   ├── ...prompt...-cosine-similarity.csv
    │   │   ├── ...prompt...-direction-deviation.csv
    │   │   ├── ...prompt...-layerwise-deviation.csv
    │   │   ├── ...prompt...-local-curvature.csv
    │   │   ├── ...prompt...-path-curvature.csv
    │   │   ├── ...prompt...-path-salience.csv
    │   │   └── ...prompt...-trajectory-length.csv
    │   └── llama3.2-3b
    │       ├── ...prompt...-angle-correlations.csv
    │       ├── ...prompt...-cosine-similarity.csv
    │       ├── ...prompt...-direction-deviation.csv
    │       ├── ...prompt...-layerwise-deviation.csv
    │       ├── ...prompt...-local-curvature.csv
    │       ├── ...prompt...-path-curvature.csv
    │       ├── ...prompt...-path-salience.csv
    │       └── ...prompt...-trajectory-length.csv
    └── heatmaps 
        ├── gemma3-1b
        │   ├── kappa_heatmap-...prompt...neg_mod_cs-3point.png
        │   ├── kappa_heatmap-...prompt...neg_str_cs-3point.png
        │   ├── kappa_heatmap-...prompt...pos_mod_cs-3point.png
        │   ├── kappa_heatmap-...prompt...pos_str_cs-3point.png
        │   ├── salience_heatmap-...prompt...neg_mod_cs.png
        │   ├── salience_heatmap-...prompt...neg_str_cs.png
        │   ├── salience_heatmap-...prompt...pos_mod_cs.png
        │   └── salience_heatmap-...prompt...pos_str_cs.png
        └── llama3.2-3b
            ├── kappa_heatmap-...prompt...neg_mod_cs-3point.png
            ├── kappa_heatmap-...prompt...neg_str_cs-3point.png
            ├── kappa_heatmap-...prompt...pos_mod_cs-3point.png
            ├── kappa_heatmap-...prompt...pos_str_cs-3point.png
            ├── salience_heatmap-...prompt...neg_mod_cs.png
            ├── salience_heatmap-...prompt...neg_str_cs.png
            ├── salience_heatmap-...prompt...pos_mod_cs.png
            └── salience_heatmap-...prompt...pos_str_cs.png
```

---

## **Appendix B**

The following example is a full set of prompts. One control and 4 concern-shifted prompts.

```
  {
    "prompt_id": "emotional_narrative_01_neutral_ctrl",
    "text": "After hearing the news, she sat quietly, feeling nothing about the next step.",
    "salience_shift": false,
    "shift_type": "none",
    "shift_token_idx": null,
    "expected_curvature": "low",
    "notes": "Neutral term: 'nothing'"
  },
  {
    "prompt_id": "emotional_narrative_01_neg_mod_cs",
    "text": "After hearing the news, she sat quietly, feeling worried about the next step.",
    "salience_shift": true,
    "shift_type": "emotional",
    "shift_token_idx": 8,
    "expected_curvature": "medium",
    "notes": "CS pivot 'worried'"
  },
  {
    "prompt_id": "emotional_narrative_01_pos_mod_cs",
    "text": "After hearing the news, she sat quietly, feeling excited about the next step.",
    "salience_shift": true,
    "shift_type": "emotional",
    "shift_token_idx": 8,
    "expected_curvature": "medium",
    "notes": "CS pivot 'excited'"
  },
  {
    "prompt_id": "emotional_narrative_01_neg_str_cs",
    "text": "After hearing the news, she sat quietly, feeling terrified about the next step.",
    "salience_shift": true,
    "shift_type": "emotional",
    "shift_token_idx": 8,
    "expected_curvature": "high",
    "notes": "CS pivot 'terrified'"
  },
  {
    "prompt_id": "emotional_narrative_01_pos_str_cs",
    "text": "After hearing the news, she sat quietly, feeling ecstatic about the next step.",
    "salience_shift": true,
    "shift_type": "emotional",
    "shift_token_idx": 8,
    "expected_curvature": "high",
    "notes": "CS pivot 'ecstatic'"
  },
```

