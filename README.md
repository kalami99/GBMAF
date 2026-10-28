**GBMAF: A Magnetic Attractive-Field-Based Granular-Ball Method for Classification Rule Generation**

This repository provides the core implementation, benchmark datasets, and experimental comparison results for GBMAF.

## Overview

GBMAF is a coarse-to-fine granular-ball computing framework for classification rule generation. It consists of three main components:

1. **Adaptive Coarse Granular-Ball Generation (ACBG):** Constructs coarse granular balls using density-guided seeds and different-class distance constraints.
2. **Magnetic-Attraction-Inspired Refinement:** Selectively refines granular balls in different-class overlap regions using attraction-based priorities.
3. **GBMAF-CV Classification:** Combines coverage-based collaborative voting and adaptive distance-weighted voting using multi-granularity rules.

## Repository Contents

| File | Description |
|---|---|
| `gbmf_methods.py` | Core GBMAF generation and classification implementation |
| `requirements.txt` | Required Python packages |
| `dataset.zip` | Benchmark datasets |
| `comparison_results_allmethods.zip` | Experimental comparison results |

## Requirements

The implementation is written in Python.

Install the required dependencies using:

`pip install -r requirements.txt`

## Parameter Configuration

The provided implementation uses a sample-size-based parameter configuration for alpha, beta, gamma, minimum granular-ball size, and maximum granular-ball count.

**Reproducibility note:** The experimental results reported in the manuscript were obtained after parameter tuning. Results obtained using the released code may differ slightly from the reported values depending on the parameter settings.

## Experimental Results

The manuscript evaluates GBMAF on 15 benchmark datasets, including granular-ball generation quality, classification performance, computational efficiency, parameter sensitivity, noise robustness, and ablation studies.

The available comparison results are provided in `comparison_results_allmethods.zip`.

## Citation

If you use this implementation in your research, please cite the corresponding paper:

**GBMAF: A Magnetic Attractive-Field-Based Granular-Ball Method for Classification Rule Generation.**

Full bibliographic information will be added when available.
