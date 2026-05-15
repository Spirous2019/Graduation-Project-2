# Graduation Project 2 — Machine Learning Techniques for PAPR Reduction in Multicarrier Communication Systems

**Ain Shams University · Faculty of Engineering · Electronics and Communications Engineering Department**
**Academic Year 2025/2026 · Supervisor: Dr. Michael Ibrahim**

---

## Overview

This repository contains the full documentation, simulation code, and research materials for the **second part** of a two-semester graduation project.

**Part I** (previous semester) established the theoretical and simulation baseline for Peak-to-Average Power Ratio (PAPR) reduction in OFDM systems, covering four classical techniques:

| Technique | Key Idea |
|---|---|
| Amplitude Clipping & Filtering | Hard-limit signal peaks; re-filter distortion |
| Tone Reservation | Reserve subcarriers to generate a cancelling peak signal |
| Selected Mapping (SLM) | Transmit the candidate with lowest PAPR from $U$ phase-rotated variants |
| Partial Transmit Sequence (PTS) | Partition subcarriers into sub-blocks; optimise per-block phase factors |

**Part II (this repository)** applies **machine learning** to overcome the fundamental limitations of those classical methods — high computational complexity, mandatory side information overhead, and exhaustive candidate search — by replacing hand-designed algorithms with end-to-end learned mappings.

---

## Project Team

| Name | Name |
|---|---|
| Nour-Eldin Mohamed Mostafa | Youssef Samir Sleem Ahmed |
| Muhammad Mahmoud Muhammad | Reem Nader Saeed |
| Basel Mohamed Ramadan | Ahmed Walid Hassan |
| Yosef Ahmed Mohamed | |

---

## Current Content

### Chapter 1 — OFDM & SLM Refresher (`Thesis/Chapters/01_OFDM_SLM_Refresher.tex`)
A self-contained, mathematically complete refresher bridging Part I and Part II:
- OFDM signal model, subcarrier mapping, IFFT-based modulation
- PAPR definition, CCDF as the standard performance metric
- SLM algorithm in full detail: phase vector generation, candidate selection, side information problem
- Why SLM's three limitations motivate the deep-learning approach

### Chapter 2 — PRNet: Deep Autoencoder for PAPR Reduction (`Thesis/Chapters/03_PRNet.tex`)
Full exposition of **PRNet** (Kim et al., 2017) — the first end-to-end deep-learning system for joint PAPR and BER optimisation in OFDM:
- Autoencoder architecture: DNN encoder (transmitter) + AWGN channel + DNN decoder (receiver)
- Joint loss function: $\mathcal{L} = \text{MSE} + \lambda \cdot \text{PAPR penalty}$
- Two-stage training procedure: BER-first pre-training, then joint BER+PAPR fine-tuning
- Corruption-based channel simulation during training
- Simulation results: BER vs SNR, PAPR CCDF, corruption-level sweep, $\lambda$ trade-off curves
- Complexity analysis, common misconceptions, limitations, and connections to subsequent work

---

## Planned Chapters *(to be added)*

| # | Chapter | Topic |
|---|---|---|
| 3 | `04_NN_SCF` | Neural Network Signal Cancellation Framework |
| 4 | `05_GA_SLM` | Genetic Algorithm-Optimised SLM |
| 5 | `06_ESLM_AE` | Extended SLM via Autoencoder (Hao et al., 2019) |
| 6 | `07_PR_DUN` | PAPR Reduction via Deep Unfolding Networks |
| 7 | `08_DL_AE_CO_OFDM` | Deep Learning Autoencoder for Coherent Optical OFDM |
| 8 | `09_Comparative_Analysis` | Side-by-side comparison of all ML methods |
| 9 | `10_Future_Directions` | Open problems and research outlook |

---

## Repository Structure

```
Graduation Project 2/
│
├── Thesis/                         # LaTeX thesis source
│   ├── Thesis.tex                  # Master document
│   ├── Preamble.tex                # Packages, styles, custom environments
│   ├── References.bib              # BibLaTeX bibliography
│   ├── Figures/                    # All figures (logo, plots, diagrams)
│   ├── Chapters/
│   │   ├── 00a_Nomenclature.tex
│   │   ├── 00b_Mathematical_Symbols.tex
│   │   ├── 01_OFDM_SLM_Refresher.tex
│   │   ├── 03_PRNet.tex
│   │   └── ...                     # Future chapters
│   └── Appendices/
│       ├── A_Consolidated_Parameters.tex
│       └── B_DL_Glossary.tex
│
├── Codes/                          # Python simulation scripts
│   └── ...
│
├── References/                     # Downloaded papers & resources
│   └── ...
│
├── ML-Enhanced SLM/               # Supplementary material / earlier drafts
│   └── ...
│
├── .gitignore
└── README.md
```

---

## Compiling the Thesis

The thesis uses **LuaLaTeX** (required for `fontspec` / OpenType fonts) and **Biber** for bibliography management.

**Recommended compile sequence:**
```bash
cd Thesis
lualatex Thesis.tex
biber Thesis
lualatex Thesis.tex
lualatex Thesis.tex
```

Or with `latexmk`:
```bash
cd Thesis
latexmk -lualatex Thesis.tex
```

**Required fonts** (must be installed on the system):
- `Source Serif 4` — main body font
- `JetBrains Mono` — monospace / code font

---

## Key References

- **Kim, K. et al. (2017).** *Deep Learning-Based PAPR Reduction Method for OFDM Systems.* — PRNet.
- **Hao, Z. et al. (2019).** *Extended SLM Autoencoder for PAPR Reduction.*
- **Alnaseri, A. et al. (2025).** *Deep Learning Autoencoder for CO-OFDM.*
- **Zou, X. et al. (2021).** *Neural Network Signal Cancellation Framework.*

---

*This project is submitted in partial fulfillment of the requirements for the degree of Bachelor of Science in Electronics and Communications Engineering, Ain Shams University.*
