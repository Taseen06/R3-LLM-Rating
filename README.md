# R3-LLM-Rating

## Title
Characterising divergence between heuristic and language-model ratings of mobile user interface quality — replication package.

## Description
This repository contains the complete data, code, and analysis pipeline supporting the manuscript submitted to PeerJ Computer Science. The study compares a rule-based heuristic scorer against three open-weight language models on the task of rating mobile user interface quality, using 15,000 screens drawn from the RICO dataset and 270,000 total ratings across two input representations (structured JSON and raw screenshot).

## Dataset Information
- **Source dataset**: RICO (Deka et al., 2017), 66,261 mobile application screens, each with a JPEG screenshot and a JSON view hierarchy. RICO is publicly available and is not redistributed in full here; only derived scores and screen identifiers are included.
- **Corpus**: 15,000 screens retained after six filtering criteria (JSON validity, minimum visible elements, clickable element presence, non-zero layout bounds, image integrity, exact-duplicate removal). See `selected_15k.csv` for the retained screen identifiers and structural features (`total_elements`, `clickable_elements`, coarse category).
- **Heuristic scores**: `heuristic_scores_15k.csv` — three dimension scores (usability, layout_quality, visual_complexity) on a 1–5 scale for every retained screen.
- **Model scores**: `llm_scores_15k_(MODEL)(REPRESENTATION).csv` — one file per model (gemma3, gemma4, qwen3-vl) per input representation (JSON, JPEG), same three-dimension scoring format.

## Code Information
- `FINAL_SELECTION.py` — applies the six corpus filters to the raw RICO dataset and produces `selected_15k.csv`.
- `FINAL_HEURISTICS.py` — computes the heuristic baseline scores from JSON hierarchies.
- `LLM_PIPELINE_(gemma3).py`, `LLM_PIPELINE_(gemma4).py`, `LLM_PIPELINE_(qwen3-vl).py` — query each model through the Ollama Cloud endpoint for both input representations, with retry logic and incremental checkpointing.
- `analyze_corrected.py` — computes all reported statistics: linearly weighted Cohen's kappa, bootstrap confidence intervals, offset-corrected kappa, ICC(A,1), Bland–Altman limits of agreement, chance-baseline for the tolerance-band rule, Spearman structural-tracking correlations (Holm-corrected), cross-representation Pearson correlations (Holm-corrected), and within-rater dimension inter-correlations. Outputs five CSV tables (`table1`–`table5`) used directly in the manuscript.
- `make_peerj_figs.py` — generates all four manuscript figures (PDF) from the table outputs.

## Usage Instructions
1. Download RICO from the original source and place it under `rico_dataset/` (not included in this repository; see licensing note below).
2. Run `FINAL_SELECTION.py` to reproduce `selected_15k.csv`, or use the version already provided.
3. Run `FINAL_HEURISTICS.py` to reproduce `heuristic_scores_15k.csv`.
4. Run each `LLM_PIPELINE_*.py` script to reproduce the six model-rating CSVs (requires an Ollama Cloud endpoint and API access; costs apply).
5. Run `analyze_corrected.py` to reproduce all five result tables from the CSVs already provided.
6. Run `make_peerj_figs.py` to reproduce all four figures from the table outputs.

Steps 3–6 can be run directly on the CSVs included in this repository without repeating the RICO download or model queries.

## Requirements
- Python 3.10+
- pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib
- Install with: `pip install pandas numpy scipy scikit-learn statsmodels matplotlib`
- Model-querying scripts additionally require `requests` and an Ollama Cloud API key set as an environment variable.

## Methodology
Full methodological detail — corpus construction, heuristic scoring formulae, model querying protocol, and statistical methods — is described in the Materials and Methods section of the associated manuscript. In summary: screens are filtered from RICO by six criteria; scored on usability, layout quality, and visual complexity by both a literature-grounded heuristic and three language models; and compared using bias-adjusted agreement statistics, a chance-corrected tolerance-band rule, and structural-tracking correlations, all computed with multiple-comparison correction where applicable.

## Citations
If you use this repository, please cite the associated manuscript (details to be added upon publication) and the original RICO dataset:

> Deka, B., Huang, Z., Franzen, C., Hibschman, J., Afergan, D., Li, Y., Nichols, J., & Kumar, R. (2017). Rico: A mobile app dataset for building data-driven design applications. *Proceedings of the 30th Annual ACM Symposium on User Interface Software and Technology*, 845–854.

## License and Contribution Guidelines
Code in this repository is released under the MIT License. Data derived from RICO (screen identifiers, structural features, heuristic and model scores) is provided for research reproducibility; the underlying RICO dataset remains subject to its own original license and is not redistributed here in full. Issues and pull requests are welcome.
