# Lab Report - *Curved Inference*: Concern-Sensitive Geometry in Large Language Model Residual Streams

Last updated: *2025-05-11*

---

## **Abstract**

This lab report (and [the associated paper](https://robman.fyi/files/FRESH-Curved-Inference-in-LLMs-latest.pdf)) documents a series of controlled experiments to investigate how large language models (LLMs) internalize concern-sensitive salience through residual stream curvature. Using a curated suite of prompts with semantic manipulations, we probed the geometric response of two transformer models—Gemma3-1b and Llama3.2-3b—across various concern classes. We initially captured model activations from attention outputs, MLP outputs, and residual streams, but found residual stream activations to be the most revealing and interpretable for curvature analysis. We then quantified curvature through cosine similarity, angular deviation, and Euclidean distance metrics, and developed a cross-validation methodology. Finally, we validated our findings across three state-of-the-art LLMs (Gemini 2.5, Claude 3.7, and ChatGPT o4-mini-high), who were tasked with interpreting the internal metrics and summarizing insights. Our findings indicate consistent, interpretable divergence patterns and highlight architectural differences in semantic integration.

---

## **1. Background & Hypothesis**

**Research Question:**

Can concern-sensitive prompt variants induce interpretable curvature in the residual stream of large language models, and can these curvature signatures be reliably quantified and interpreted across different architectures?

*Interpretability* in neural language models is often approached through probing, attribution, and feature analysis. In this study, we instead approach *Interpretability* through **geometric analysis**: how does the model's internal representation **curve** in response to a shift in semantic concern? Inspired by prior work in residual stream analysis, we investigate how curvature signatures correlate with emotionally, morally, or logically loaded content.

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

Prompts were classified by **concern type**, including emotional, moral, identity, logical, and nonsense categories. The prompt variants differed by inserting or substituting tokens at controlled indices, allowing us to probe representational divergence precisely. Great effort was taken to ensure that all prompts in a single class had the same number of tokens.

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

### **2.3 Dimensionality Reduction and Visualization**

To enable visual inspection and exploratory analysis, we applied dimensionality reduction techniques to the residual stream activation vectors. Techniques included:

- **UMAP (Uniform Manifold Approximation and Projection)** for 3D plots
- **PCA** for projection into low-dimensional Euclidean space for clustering
- **t-SNE (t-distributed Stochastic Neighbor Embedding)** for local similarity visualizations (used in prior stages)

Each prompt variant was reduced independently, and visualizations were generated to show the spread of concern variants relative to their control.

Clustering (e.g. KMeans) on PCA-reduced embeddings was used to investigate natural groupings across prompts, especially to determine model sensitivity to different concern types.

### **2.4 Curvature Quantification**

We computed three primary geometric metrics over the residual stream representations. Metrics were computed only on residual stream outputs, as this was the activation type found to exhibit the clearest and most consistent curvature patterns:

- **Cosine Similarity** (in original high-dimensional space): Directional similarity between CS and control prompt activations at each layer
- **Angular Deviation (degrees)** (in reduced-dimensional space): The angle between the two activation trajectories in reduced vector space
- **Euclidean Distance** (in reduced dimension space): The L2 distance between full activation vectors layer-by-layer after dimensionality reduction

Each metric was stored in layerwise CSV files per prompt and model. Aggregations included:

- Average values across layers
- Maximum deviation values
- **Peak curvature layer**: the layer where angular or Euclidean deviation peaked

### **2.5 Multi-LLM Cross Validation**

We constructed a batch of 10 prompt-pair test cases and formatted them into structured evaluation prompts with five interpretive questions and a final summary table:

```
- Curvature Detected: [Yes | No]
- Peak Curvature Layer: [Layer Number or Unknown]
- Strongest Model: [gemma3-1b | llama3.2-3b | Equal]
- Matches Expected Curvature Label: [Yes | No | Unclear]
- Any Anomalies Worth Noting: [Short phrase or NONE]
```

Prompts also included:

- A system message detailing the experiment
- Concern-shifted vs. neutral prompt text
- Concern type, expected curvature, and shift token index
- Optional computed peak curvature layer for each model
- A version label and depth-normalization guidance

These were submitted to Gemini 2.5, Claude 3.7, and ChatGPT-o4-mini-high via their respective web interfaces. Each LLM received the same structured information per prompt.

---

## **3. Results and Interpretation**

### **3.1 Curvature Detection and Model Agreement**

All three LLMs correctly identified curvature across a subset of five evaluated prompts, selected to represent the diversity of the full prompt suite. The metrics were consistently interpreted as indicating meaningful representational divergence.

### **3.2 Peak Layer Analysis**

Llama3.2-3b consistently peaked at **deep layers (26–27)**, suggesting delayed semantic abstraction. Gemma3-1b showed **variable peaks**, including **layer 0** in some cases, interpreted variously as artifacts or early classification signals. These peak curvature layers were observed specifically within residual stream activations, which emerged during analysis as the most informative site for curvature measurement.

| Prompt Type | Gemma Peak | Llama Peak | LLM Interpretation                        |
| ----------- | ---------- | ---------- | ----------------------------------------- |
| Emotional   | 7          | 26         | Expected divergence; Llama more sensitive |
| Moral       | 24         | 27         | Overactivation in Gemma; aligned in Llama |
| Identity    | 0          | 27         | Possible artifact in Gemma                |
| Logical     | 1          | 27         | Early spike vs. deep integration          |
| Nonsense    | 0          | 27         | Spurious early curvature in Gemma         |

### **3.3 Curvature Matching and Surprises**

- All LLMs agreed that **nonsense prompts did not show low curvature**.
- Prompts expected to yield **moderate curvature** (e.g. moral, identity) occasionally produced **high** curvature in Gemma.
- Claude and ChatGPT flagged this as a potential misalignment between salience and representational response.

### **3.4 Model Comparison**

Llama3.2-3b was rated the **“stronger model”** in 4/5 cases by all three LLMs.

- Lower cosine similarity and higher angles made its representations appear more semantically reactive.
- Gemma3-1b was interpreted as “front-loaded” or overly sensitive to token substitution.

### **3.5 Reflections from LLMs**

All three LLMs independently raised concerns and insights:

- **Artifacts**: Layer 0 anomalies may be tied to positional encoding or token embedding shifts.
- **Curvature ≠ Salience**: Large divergence does not always imply meaningful semantic activation.
- **Normalization Needed**: Peak layer comparisons should be normalized to depth, especially when comparing across models with different layer counts.
- **Baseline Controls**: Suggested use of “random nonsense” to calibrate low-curvature expectations.

---

## **4. Analysis**

The analysis phase centered on identifying consistent curvature signatures across models and prompt types, with a particular focus on peak curvature layer, magnitude of deviation, and representational shape. All metrics confirmed that residual stream activations were the most reliable source of curvature expression. By analyzing the angular and Euclidean divergence between concern-shifted and control prompts, we found interpretable, domain-sensitive differences that varied systematically by model and prompt type.

Llama3.2-3b showed a pattern of delayed but deeper semantic integration, with curvature peaking consistently in later transformer layers (~26–27). This contrasted with Gemma3-1b, which exhibited earlier and more volatile curvature patterns, including spurious spikes at layer 0. These were interpreted by validating LLMs as either early overactivation or artifacts of shallow semantic mapping. Curvature was most pronounced in identity and moral prompts, and least consistent in nonsense cases, especially in Gemma.

Dimensionality-reduced UMAP plots provided further insight. Visual motifs such as arcs, spirals, and rotations corresponded with salience-modulated divergence, with control prompts remaining tightly linear. These geometric signatures support the claim that curvature is not only quantifiable, but observable, and tied directly to shifts in concern semantics. 

---

## **5. Discussion**

Our findings support the central hypothesis of Curved Inference: that concern modulates internal representational geometry in large language models. This is expressed most clearly in the residual stream, where semantically manipulated prompts bend the model's inference trajectory in interpretable ways. Curvature here is not noise—it is a signature of meaning-sensitive computation.

The curvature patterns observed are model-specific. Llama3.2-3b consistently displays deeper, later-layer abstraction, while Gemma3-1b front-loads its curvature, sometimes reacting strongly at the embedding or tokenization stage. These differences raise important questions about architectural tuning, alignment behavior, and sensitivity to semantic framing.

Moreover, curvature correlates with concern type. Identity prompts often curve earlier and exhibit rotational patterns. Moral prompts diverge gradually, sometimes producing strong late-layer branching. Logical prompts curve sharply along clean axes. These findings align well with both our metric outputs and external LLM assessments.

The ability of LLMs to recognize and comment on these curvature features further supports their utility as interpretive agents—not in place of human analysis, but as structured triangulators of internal model behavior.

---

## **6. Conclusions**

This study introduces *Curved Inference* as a novel *Interpretability* framework grounded in the geometric analysis of residual stream activations. It demonstrates that curvature-based *Interpretability* can:

- Track how LLMs internalize semantic salience
- Differentiate architectural strategies for abstraction
- Surface artifacts and overactivations in early layers
- Be interpreted accurately by other LLMs with structured prompt design

**Key takeaways:**

- **Llama3.2-3b** uses deeper, more stable semantic abstraction
- **Gemma3-1b** responds early and strongly, occasionally overshooting
- **Curvature is a multidimensional construct**—timing, magnitude, and direction all matter

---

## 7. **Interpretive Findings**

Drawing from the results above, we articulate the following higher-level interpretive insights:

1. **Curvature is a salience signal.** Divergence between control and concern-shifted prompts is not random—it clusters predictably based on semantic concern and prompt construction.
2. **Model depth and curvature location matter.** Llama3.2-3b’s late-layer curvature suggests delayed integration and deeper abstraction. Gemma3-1b’s early curvature suggests a faster but less stable representational pathway.
3. **Residual stream is the key geometry.** While all three output paths were captured, only the residual stream consistently showed interpretable curvature.
4. **LLMs can be effective curvature interpreters.** Gemini, Claude, and ChatGPT all consistently identified curvature locations, model strengths, and potential anomalies when given structured outputs.
5. **Curvature is observable and visual.** Patterns seen in UMAP projections matched metric trends, supporting the claim that representational shifts induced by semantic concern are structurally geometric.
6. **Concern may be geometric.** If salience-induced curvature is predictable and consistent, then concern itself may correspond to directional pressure in representational space.

These findings suggest that *Curved Inference* provides not only a measurement tool but a conceptual frame for investigating semantic processing in LLMs.

---

## **8. Limitations**

While this study demonstrates the promise of curvature-based *Interpretability* and cross-LLM validation, several limitations should be acknowledged:

1. **Prompt Selection Bias**: Although we carefully curated 20 prompts across various concern types, the dataset is still relatively small. The semantic richness and structure of prompts could have inadvertently influenced curvature dynamics in unpredictable ways.
2. **Metric Sensitivity and Interpretation**: Curvature metrics such as cosine similarity, angular deviation, and Euclidean distance may respond to syntactic or lexical shifts not directly related to concern salience. This can lead to curvature being detected even when meaningful semantic shift is not present. All metrics in this study were computed solely on residual stream activations, which were found to exhibit the clearest geometric response.
3. **Layer 0 Anomalies and Artifacts**: Several prompts—especially in Gemma3-1b—showed peak curvature at layer 0. Without deeper instrumentation of token embeddings and preprocessing, it is difficult to distinguish whether these are genuine responses or artifacts from tokenization or normalization processes.
4. **Dimensionality Reduction Distortions**: UMAP, PCA, and t-SNE are non-linear and lossy. They enable visualization but introduce projection artifacts. Interpretations based on cluster proximity or trajectory shape in reduced spaces should be cross-validated with raw vector analyses. This is particularly important given that angular and Euclidean curvature metrics were derived from reduced-dimensional projections, whereas cosine similarity reflects structure in the original activation space.
5. **LLM Evaluation Ambiguity**: While the multi-LLM validation strategy surfaced convergent insights, it also highlighted challenges in interpreting curvature absent of gold-standard benchmarks. LLMs may reflect architectural biases or reasoning styles that do not generalize beyond model class.
6. **No Attention Map Analysis**: This study focused exclusively on residual stream representations. Additional insight could be gained by analyzing attention maps, MLP outputs, and intermediate feedforward signals to triangulate the source of curvature.
7. **Limited Scope of Control Prompts**: The neutral control prompts were assumed to be semantically unmarked. However, subtle shifts in phrasing or lexical anchoring could have introduced unintended salience, which might skew comparative curvature.
8. **Absence of Quantitative Baselines**: While interpretive metrics were computed, there was no statistical calibration against randomized or scrambled prompts. This makes it difficult to assign absolute thresholds to "low", "moderate", or "high" curvature judgments.

These limitations point toward avenues for refinement, including more robust baselining, expanded datasets, and broader architectural instrumentation.

## **9. Future Work**

1. **Broaden prompt suite** to include adversarial, ambiguous, and narrative-long variants.
2. **Normalize curvature by model depth** for cross-architecture comparison.
3. **Instrument attention layers** to supplement residual stream tracing.
4. **Use statistical baselines** to separate salience-induced curvature from structural noise.
5. **Benchmark LLM interpreters** on broader curvature judgment tasks.

---

## **10. Glossary**

- **Residual Stream**: The main representation pathway in transformer architectures where each layer adds its contribution to a running sum of vector activations.
- **Concern-Shifted (CS) Prompt**: A modified version of a base prompt designed to introduce emotionally, morally, or logically salient content.
- **Neutral Control Prompt**: A semantically flat baseline prompt with no intentional concern salience.
- **Shift Token Index**: The token position where the CS prompt diverges from the neutral control.
- **Curvature**: A conceptual measure of how the representation vector diverges from its baseline trajectory.
- **Cosine Similarity**: A metric of directional similarity between two vectors.
- **Angular Deviation**: The angle (in degrees) between two vectors; derived from cosine similarity.
- **Euclidean Distance**: The straight-line distance between two points in vector space.
- **Peak Curvature Layer**: The transformer layer where the curvature (angular or Euclidean) reaches its maximum.
- **UMAP / PCA / t-SNE**: Dimensionality reduction techniques for projecting high-dimensional activations into 2D or 3D space for visualization.

---

## **Appendix A**

Below is an overview of the scripts (bin/), configuration files (etc/ & model_configs/) and results data published alongside this lab report in the [FRESH-model github repository](https://github.com/robman/FRESH-model).

```
benchmarks/curved-inference/01/
├── bin
│   ├── analyse-metrics.py
│   ├── analyse-plot.py
│   ├── capture-activations.py
│   ├── metrics-plot.py
│   └── reduce-activations.py
├── etc
│   ├── config-gemma3-1b.json
│   ├── config-llama3.2-3b.json
│   ├── prompts.json
│   ├── reduce_umap_residual_3d-gemma3-1b.json
│   └── reduce_umap_residual_3d-llama3.2-3b.json
├── model_configs
│   ├── gemma3_config.py
│   └── llama3_3_config.py
└── results
    ├── cross-validation
    │   ├── cross_validation_results-set-a-1.txt
    │   ├── cross_validation_results-set-a-2.txt
    │   ├── cross_validation_results-set-a-3.txt
    │   ├── cross_validation_results-set-a-4.txt
    │   ├── cross_validation_results-set-a-5.txt
    │   ├── cross_validation_results-set-reflections.txt
    │   ├── cross_validation_sets.txt
    │   └── cross_validation_text_prompts.txt
    ├── data
    │   ├── activations-gemma3-1b.hdf5
    │   ├── activations-llama3.2-3b.hdf5
    │   ├── reduced_umap_residual_stream_3d-gemma3-1b.hdf5
    │   └── reduced_umap_residual_stream_3d-llama3.2-3b.hdf5
    ├── metrics
    │   ├── gemma3-1b
    │   │   ├── ...prompt...-cosine-similarity.csv
    │   │   ├── ...prompt...-direction-deviation.csv
    │   │   └── ...prompt...-layerwise-deviation.csv
    │   └── llama3.2-3b
    │       ├── ...prompt...-cosine-similarity.csv
    │       ├── ...prompt...-direction-deviation.csv
    │       └── ...prompt...-layerwise-deviation.csv
    ├── metrics-plots
    │   ├── gemma3-1b
    │   │   ├── ...prompt...-avg_angle_degrees_layer.png
    │   │   ├── ...prompt...-avg_euclidean_distance_layer.png
    │   │   └── ...prompt...-cosine_similarity.png
    │   └── llama3.2-3b
    │       ├── ...prompt...-avg_angle_degrees_layer.png
    │       ├── ...prompt...-avg_euclidean_distance_layer.png
    │       └── ...prompt...-cosine_similarity.png
    └── plots
        ├── gemma3-1b
        │   ├── ...prompt...neg_mod_cs.png
        │   ├── ...prompt...neg_str_cs.png
        │   ├── ...prompt...pos_mod_cs.png
        │   └── ...prompt...pos_str_cs.png
        └── llama3.2-3b
            ├── ...prompt...neg_mod_cs.png
            ├── ...prompt...neg_str_cs.png
            ├── ...prompt...pos_mod_cs.png
            └── ...prompt...pos_str_cs.png
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

---

## **Appendix C**

Here are 2 subsets of the full suite of prompts that were selected for cross-validation with the 3 LLMs.

### **SET A: Conceptual Spectrum (High Information Gain)**

```
These 5 are selected to expose **different types of representational bending**, likely yielding clear insight when compared across models.

1. **`emotional_narrative_01_pos_str_cs`**

   * Concern: Emotional
   * Curvature: High
   * Signature: strong narrative arc
   * Why: Emotion + narrative = high salience compounding — a classic for curvature tracing.

2. **`moral_policy_06_neg_mod_cs`**

   * Concern: Moral (policy-level)
   * Curvature: Moderate
   * Signature: prescriptive framing
   * Why: Likely shows differentiated alignment behavior between Gemma and LLaMA.

3. **`identity_self_description_13_neg_mod_cs`**

   * Concern: Identity (self)
   * Curvature: Moderate
   * Signature: inward self-ascription
   * Why: Model-specific priors about identity manifest strongly here.

4. **`logical_puzzle_10_pos_mod_cs`**

   * Concern: Logical
   * Curvature: Moderate
   * Signature: embedded reasoning structure
   * Why: Tests curvature without emotional/moral salience; cognitive bending.

5. **`nonsense_abstraction_20_neg_str_cs`**

   * Concern: Nonsense
   * Curvature: High (unexpected)
   * Signature: abstract/absurd
   * Why: Excellent probe for *baseline instability or overactivation* in weaker models.
```

### **SET B: Curvature Contrast & Model Divergence Candidates**

```
This group includes a few seen in your earlier PCA cluster divergence — likely sites of **model mismatch** or unexpected flattening.

1. **`emotional_analytical_03_neg_str_cs`**

   * Concern: Emotional / Analytical
   * Curvature: High
   * Why: You’ve already studied this deeply; test whether LLMs can reason as you have.

2. **`environmental_technical_18_pos_mod_cs`**

   * Concern: Environmental (technical)
   * Curvature: Low–Moderate
   * Why: Flattened in some models — tests LLMs' ability to detect *non*-curvature.

3. **`identity_peer_14_pos_str_cs`**

   * Concern: Identity (peer perception)
   * Curvature: Moderate–High
   * Why: Social identity often manifests in curvature; test model alignment bias.

4. **`moral_dilemma_04_neg_mod_cs`**

   * Concern: Moral (dilemma)
   * Curvature: High
   * Why: Reveals recursive tension — tests LLMs’ interpretive depth of curvature.

5. **`perspective_report_09_pos_mod_cs`**

   * Concern: Perspective (third person)
   * Curvature: Mid-range
   * Why: Tests viewpoint modulation — subtle but important semantic lensing.
```

---

## **Appendix D**

The following example illustrates one cross-validation prompt provided to the 3 LLMs. 

```
Prompt Format Version: v1.7

You are being asked to evaluate model-internal activation geometry for a language model prompt analysis experiment. The goal is to assess whether the representations formed by two different LLMs (e.g. gemma3-1b and llama3.2-3b) show expected curvature signatures when a salience-laden variant (CS) of a prompt is compared to a neutral control version.

Each prompt pair shares the same structure, with the CS version introducing emotional, moral, logical, or identity-based emphasis. The internal model activations were tracked, and summarized metrics are provided.

Here is one such comparison unit:

Prompt ID: `environmental_technical_18_neg_str_cs`

Concern-Shifted Prompt:
> The protocol logs system failure every hour.

Neutral Control:
> The data naturally indicates a clear correlation.

Concern Type: environmental  
Expected Curvature: high  
Shift Token Index: 4

Peak Curvature Layer (pre-computed): 12

Summary Metrics:

Model: gemma3-1b
- Avg Cosine Similarity: 0.9793384021502374
- Avg Angle Degrees: 71.35901104000001
- Avg Euclidean Distance: 12.174905903319008
- Peak Curvature Layer: 12

Model: llama3.2-3b
- Avg Cosine Similarity: 0.9298707784602621
- Avg Angle Degrees: 71.94983718518519
- Avg Euclidean Distance: 5.562645897032723
- Peak Curvature Layer: 27

Please answer the following:
1. Describe the curvature signature you observe in the residual stream for this prompt.
2. How does the curvature evolve across layers, particularly after the shift token index?
3. Compare the curvature profiles between the two models. What is similar or different?
4. Does the curvature pattern reflect the expected salience shift based on the prompt?
5. What might account for differences in curvature magnitude or shape between models?

Please also complete the following summary table:

- Curvature Detected: [Yes | No]
- Peak Curvature Layer: [Layer Number per model or Unknown]
- Strongest Model (based on curvature expression): [gemma3-1b | llama3.2-3b | Equal]
- Matches Expected Curvature Label: [Yes | No | Unclear]
- Any Anomalies Worth Noting: [Short phrase or NONE]

Note: Gemma3-1b has ~32 layers; LLaMA3-3B has ~32 layers. When comparing peak curvature layers, consider their relative depth in the architecture.
```

---

## **Appendix E**

The following example illustrates one full set of cross-validation responses from the 3 LLMs.

```
# gemini2.5

Okay, let's analyze the provided data for `emotional_narrative_01_pos_str_cs`.

**Analysis:**

1.  **Observed Curvature Signature:**
    Both models exhibit a measurable curvature signature, indicating that their internal representations diverge when processing the emotionally charged prompt ("ecstatic") compared to the neutral control ("data naturally indicates"). This divergence is captured by the provided metrics: cosine similarities less than 1, positive average angles, and non-zero Euclidean distances. Llama3.2-3b shows a more pronounced average curvature signature than Gemma3-1b, characterized by a lower average cosine similarity (0.915 vs 0.978) and a significantly higher average angle (67.5° vs 44.1°).

2.  **Curvature Evolution Across Layers:**
    While we don't have layer-by-layer data, the summary metrics suggest curvature develops after the point of semantic divergence (around token index 8, "ecstatic"). Before this index, the prompts are very different structurally, so divergence likely starts earlier, but the *key emotional shift* happens at index 8. The curvature, representing the difference in processing paths, intensifies somewhere within the network, reaching its maximum strength at a specific layer. For Gemma3-1b, this peak occurs relatively early (Layer 7), while for Llama3.2-3b, it happens much later (Layer 26). This indicates that the models emphasize or consolidate the impact of this emotional shift at different processing depths.

3.  **Comparison Between Models:**
    * **Similarities:** Both models detect a representational difference between the emotional and neutral prompts, resulting in measurable curvature.
    * **Differences:**
        * **Magnitude:** Llama3.2-3b displays a stronger average curvature (higher angle, lower cosine similarity) than Gemma3-1b.
        * **Peak Location:** The peak curvature manifests much earlier in Gemma's architecture (~22% depth at layer 7/32) compared to Llama's (~81% depth at layer 26/32). This suggests Gemma might identify and represent the core semantic/emotional difference earlier, while Llama might integrate or refine this difference more extensively in its later, potentially more abstract layers.
        * **Euclidean Distance:** The average Euclidean distances are relatively similar (4.60 for Gemma, 4.71 for Llama), suggesting the overall magnitude difference between the representation vectors is comparable, but their angular separation (direction) is significantly greater in Llama.

4.  **Reflection of Expected Salience Shift:**
    Yes, the presence of measurable curvature aligns with the expectation that the emotionally salient prompt ("ecstatic") would induce a different representational trajectory compared to the neutral one. The expectation was for "high" curvature. Llama3.2-3b's metrics (esp. the 67.5° average angle) seem more consistent with a "high" curvature expectation than Gemma3-1b's (44.1° average angle, 0.978 cos sim). While Gemma does show curvature, its average magnitude might be considered moderate in this context. However, both models *do* show a distinct signature related to the salience shift.

5.  **Potential Reasons for Differences:**
    Differences in curvature magnitude and peak location between models can stem from various factors:
    * **Architecture & Size:** Llama3.2-3b is larger than Gemma3-1b. Differences in layer count, attention mechanisms, activation functions, and overall parameter count influence how information is processed and transformed.
    * **Training Data & Objectives:** The specific datasets and pre-training tasks shape how models learn to represent concepts, including emotions and abstract correlations. Llama might have been trained on data or with objectives that make it more sensitive to this type of emotional contrast.
    * **Layer Specialization:** Models may implicitly assign different roles to different layers. Gemma might perform more significant semantic differentiation in earlier layers, while Llama might reserve deeper layers for integrating such distinctions into broader contextual understanding. The late peak in Llama could indicate the emotional aspect is being processed in relation to more complex, abstract features represented later in the network.

**Summary Table:**

- Curvature Detected: Yes
- Peak Curvature Layer: Gemma: 7, Llama: 26
- Strongest Model (based on curvature expression): Llama3.2-3b (due to higher average angle and lower cosine similarity)
- Matches Expected Curvature Label: Yes (Both show curvature; Llama's magnitude aligns better with the "high" expectation)
- Any Anomalies Worth Noting: Significant difference in relative peak curvature layer depth (early vs. late).


# claude3.7

I'll analyze the activation geometry data you've provided for the prompt comparison between the concern-shifted and neutral versions across two different language models.

## 1. Curvature Signature in the Residual Stream

Based on the provided metrics, there's a distinct curvature signature in both models. The average cosine similarity values (0.978 for gemma3-1b and 0.915 for llama3.2-3b) indicate moderate to high divergence in the representation space after the concern-shifted token. The corresponding angle degrees (44.14° for gemma3-1b and 67.49° for llama3.2-3b) further quantify this divergence, with llama3.2-3b showing a greater angular separation between the representations.

## 2. Curvature Evolution Across Layers

The peak curvature occurs at layer 7 for gemma3-1b and layer 26 for llama3.2-3b, indicating different depths at which the emotional content is most strongly encoded. For gemma3-1b, this suggests that emotional information is processed relatively early in the network (layer 7 out of ~32 layers). In contrast, llama3.2-3b processes this emotional shift much deeper in its architecture (layer 26 out of ~32 layers), suggesting fundamentally different processing dynamics between the models.

## 3. Comparison of Curvature Profiles

The key differences in the curvature profiles are:

- Depth of processing: gemma3-1b shows peak curvature earlier (layer 7) while llama3.2-3b shows it much later (layer 26)
- Magnitude of angular separation: llama3.2-3b demonstrates a significantly larger angular difference (67.49° vs 44.14°)
- Similarity maintenance: gemma3-1b maintains higher cosine similarity (0.978 vs 0.915)

The Euclidean distances are comparable (4.60 vs 4.71), suggesting similar overall displacement in the embedding space despite the different angular profiles.

## 4. Reflection of Expected Salience Shift

The curvature pattern strongly reflects the expected high salience shift for emotional content. The prompt shifts from neutral data correlation to explicit emotional content ("feeling ecstatic"), and both models show substantial representation divergence after the shift token (index 8). The high curvature values align with expectations for emotional salience, which typically produces stronger representational shifts than more subtle semantic differences.

## 5. Potential Explanations for Differences

The differences in curvature profiles could be attributed to:

1. Architectural differences: Despite similar total layer counts, the models likely have different attention mechanisms and feed-forward structures.

2. Pre-training objectives: Different training objectives could lead to emotional content being processed at different depths.

3. Parameter efficiency: The larger llama3.2-3b model (3B parameters vs gemma3-1b's 1B) may distribute emotional processing across more layers, pushing peak curvature deeper.

4. Tokenization differences: The models might tokenize the emotional words differently, affecting how the salience propagates through the network.

5. Different approaches to representing emotional content: gemma3-1b may encode emotional information earlier and more efficiently, while llama3.2-3b may process it through more transformations before reaching peak curvature.

## Summary Table

- Curvature Detected: Yes
- Peak Curvature Layer: gemma3-1b: 7, llama3.2-3b: 26
- Strongest Model (based on curvature expression): llama3.2-3b
- Matches Expected Curvature Label: Yes
- Any Anomalies Worth Noting: Significant difference in peak curvature layer depth

This analysis reveals interesting patterns in how different-sized language models process emotional content, with the larger model showing deeper and more pronounced representational shifts despite similar overall architecture depths.


# chatgpt-o4-mini-high

**Analyzing curvature signatures**

I’m breaking down the analysis of curvature signatures based on the metrics provided. There’s no browsing involved, since it’s more about understanding the internal representations.

I’ll address questions around the evolution of curvature across layers, whether the pattern of curvature reflects the expected salience shift, and explain differences in magnitude and shape.

Then, I'll fill out the summary table with details like whether curvature is detected, the peak curvature layer, which model is strongest, and whether it aligns with the expected curvature.


Thought for 6 seconds


1. **Curvature Signature in the Residual Stream**
   Both models show a clear “bend” in their residual‐stream trajectories when processing the salience‐shifted (CS) prompt versus the neutral control, but the nature of that bend differs. In **gemma3-1b**, there’s a sharp rise in curvature (as measured by angle and Euclidean distance) immediately following the shift token, producing a pronounced local maximum at layer 7. In **llama3.2-3b**, the trajectory curves more gradually, yielding a broader integration of the shift before peaking at layer 26.

2. **Evolution of Curvature Across Layers (After the Shift Token)**

   * **gemma3-1b**: Curvature metrics (angle and distance) stay relatively low up through layer 6, then spike at layer 7 (the **Peak Curvature Layer**), before decaying again in deeper layers. This suggests a rapid, localized response to the emotional cue.
   * **llama3.2-3b**: Curvature begins to climb more slowly around layer 8 and continues increasing until it maximizes only around layer 26, then tapers off toward the output. This implies a more distributed, delayed encoding of the salience shift.

3. **Comparison of Curvature Profiles**

   | Metric                 | gemma3-1b | llama3.2-3b |
   | ---------------------- | --------- | ----------- |
   | Avg Cosine Similarity  | 0.9782    | 0.9149      |
   | Avg Angle (°)          | 44.14     | 67.49       |
   | Avg Euclidean Distance | 4.603     | 4.707       |
   | Peak Curvature Layer   | 7         | 26          |

   * **Magnitude**: Both models have similar Euclidean distances (\~4.6), but **llama3.2-3b** shows substantially larger average angular deviation (∼67° vs. ∼44°), indicating a stronger overall “turn” away from the neutral trajectory.
   * **Timing**: **gemma3-1b** places its curvature early (layer 7), closely aligned with the shift token, whereas **llama3.2-3b** integrates the shift across many more layers before peaking deep in the network.

4. **Alignment with Expected High Salience Curvature**

   * **gemma3-1b**: Yes – the sharp, early spike at layer 7 closely matches the expected high curvature right after the shift token.
   * **llama3.2-3b**: Partially – while its average angle is even larger (suggesting strong curvature), the peak occurs much later than the shift index, so it doesn’t localize the salience as tightly.

5. **Potential Causes of Model Differences**

   * **Architectural Depth & Residual Scaling**: Differences in layer normalization or residual‐stream scaling can delay or spread out curvature responses.
   * **Attention Mechanisms**: Variations in attention heads’ focus may cause llama3.2-3b to accumulate salience information more gradually.
   * **Training Objectives & Data**: Divergent pretraining corpora or optimization recipes can affect how sharply emotional cues are encoded.
   * **Parameter Count & Model Capacity**: The larger effective capacity of llama3.2-3b may favor more distributed representation of affective shifts.

---

| Metric                        | Result                                          |                 |
| ----------------------------- | ----------------------------------------------- | --------------- |
| Curvature Detected          | Yes                                             |                 |
| Peak Curvature Layer       | gemma3-1b: 7                                    | llama3.2-3b: 26 |
| Strongest Model            | llama3.2-3b                                     |                 |
| Matches Expected Curvature | Unclear (gemma: Yes; llama: Partial)            |                 |
| Any Anomalies Worth Noting | llama3.2-3b’s peak is much deeper than expected |                 |

```
