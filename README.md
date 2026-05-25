# Seeing Without Understanding

### Large Language Model Evaluation of Mobile UI Quality — Failure Taxonomy and Architectural Explanation

<p align="center">
  <em>Can large language models judge interface quality the way heuristics or humans do?</em><br>
  <strong>Across 15,000 screens and 270,000 ratings, the answer is: not yet — and the way they fail is architectural.</strong>
</p>

---

## Overview

This repository contains the complete replication package for **R3**, an empirical study evaluating whether large language models can reliably rate mobile user interface quality. We compare three open-weight models against a literature-grounded heuristic baseline across the full RICO dataset, then interpret the failures through transformer architectural mechanisms.

The headline result: agreement between language model judgment and a structural baseline is **statistically equivalent to chance in 17 of 18 evaluation cells**. The models do not track structural information that sits plainly in their input, they systematically underrate usability, and they give the same screen different scores depending on whether it is presented as JSON or as an image.

---

## Key Findings

| Finding | Result |
|---|---|
| **Agreement at scale** | Cohen's κ near chance in 17 of 18 cells (best: 0.227) |
| **Usability bias** | Models underrate usability by ~1 full scale point |
| **Structural decoupling** | Heuristic tracks element count at ρ = 0.94; LLMs do not |
| **Modality dependence** | Same screen, different format → different score (r = 0.00–0.18) |
| **Scale invariance** | Failure persists from 27B to 235B parameters |

---

## Research Program

R3 is the applied arm of a three-paper program on the structural origins of hallucination in large language models.

---

## Method in Brief

**Dataset.** We start from the complete RICO dataset of 66,261 real mobile app screens and filter through seven quality criteria — structural JSON validity, minimum visible elements, clickable component presence, non-zero bounds, image integrity, perceptual duplicate removal, and category derivation — retaining a clean corpus of 15,000 screens.

**Baseline.** A heuristic scorer reads each screen's JSON metadata and rates it 1–5 on usability, layout quality, and visual complexity, using severity-weighted signals, normalized layout metrics, and pixel-ratio complexity measures drawn from established HCI literature.

**Models.** Three open-weight models accessed through Ollama Cloud:

- `gemma3:27b-cloud` — 27B parameters
- `gemma4:31b-cloud` — 31B parameters
- `qwen3-vl:235b-instruct-cloud` — 235B parameters, vision-capable

**Scale.** Each model rates every screen in two modalities (structured JSON and raw screenshot), producing **270,000 individual ratings**.

---

## Reproducing the Study

```bash
# 1. Clone
git clone https://github.com/<your-username>/r3-ui-quality-evaluation.git
cd r3-ui-quality-evaluation

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl jupyter

# 3. Reproduce the analysis (uses the included data files)
jupyter notebook code/R3_Analysis.ipynb
# Run all cells → regenerates every figure and table in results/
```

To reproduce from raw data, download RICO from [interactionmining.org/rico](https://interactionmining.org/rico), then run `FINAL_SELECTION.py` → `FINAL_HEURISTICS.py` → `LLM_PIPELINE_*.py` in order. The LLM pipelines require an Ollama Cloud endpoint.

---

## The Three Architectural Mechanisms

The failure patterns map onto three mechanisms from the companion theoretical framework (R1):

**Self-attention misgrounding** — the model attends to surface co-occurrence rather than structure. Confirmed uniformly: every model fails to track element counts present in its input.

**MLE plausibility bias** — training rewards statistically likely outputs over accurate ones. Confirmed for two of three models through distributional compression; the third shows an informative exception.

**Surface pattern dependence** — outputs follow the form of the input rather than its meaning. Confirmed at the modality level: the same screen scores differently as JSON versus image.

---

## Citation

If you use this work, please cite:

```bibtex
@article{sadi2026seeing,
  title   = {Seeing Without Understanding: Large Language Model
             Evaluation of Mobile User Interface Quality,
             Failure Taxonomy, and Architectural Explanation},
  author  = {Sadi, Md Rejaul Korim and Naeem, Golam Mostofa and
             Tasin, Toufiqur Rahman and Moosa, Syed Mostofa and
             Emon, Mahmudul Hasan and Rashid, Mahmudur and
             Ahmed, Ferdus},
  year    = {2026},
  note    = {Metropolitan University, Sylhet, Bangladesh}
}
```

Companion theoretical framework:

```bibtex
@article{sadi2026architecture,
  title   = {From Architecture to Output: Structural Origins of
             Hallucination in Large Language Models and the
             Amplifying Role of Data},
  author  = {Sadi, Md Rejaul Korim and Tasin, Toufiqur Rahman and
             Naeem, Golam Mostofa},
  journal = {Available at SSRN 6604798},
  year    = {2026}
}
```

---

## Data & Dataset DOI

The full replication package is also archived on Kaggle with a citable DOI:

**https://www.kaggle.com/datasets/mdrejaulkorimsadi/r3-ui-quality-evaluation-by-llms**
DOI: [10.34740/kaggle/dsv/16150891](https://doi.org/10.34740/kaggle/dsv/16150891)

The underlying RICO dataset is available from [interactionmining.org/rico](https://interactionmining.org/rico) under its own license.

---

## Authors

Department of Computer Science and Engineering, **Metropolitan University, Sylhet, Bangladesh**

Md Rejaul Korim Sadi · Golam Mostofa Naeem · Toufiqur Rahman Tasin · Syed Mostofa Moosa · Mahmudul Hasan Emon · Mahmudur Rashid · Ferdus Ahmed

---

## License

Code is released under the **MIT License**. Data derived from RICO is subject to the original RICO dataset license. See [`LICENSE`](LICENSE) for details.

---

<p align="center">
  <em>Language models do not yet see UI quality.<br>They process it without understanding it.</em>
</p>
