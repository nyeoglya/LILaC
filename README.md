# LILaC

This repository provides a **reimplementation of LILaC**, a retrieval-augmented reasoning framework proposed in the original paper.

The goal of this project is to reproduce the core components and evaluation results of LILaC, and to empirically compare its performance against a baseline dense retriever (MM-Embed).

> This is **not** the official implementation.

## Overview

LILaC is a framework that improves multi-hop retrieval and reasoning by leveraging:
- Create a 2-Layered Component Graph
- Late Interaction on a graph
- LLM-guided query refinement

This reimplementation focuses on:
- Reproducing the retrieval pipeline
- Evaluating retrieval quality and downstream QA performance
- Reporting comparable metrics to the original paper where possible

## Evaluation Dataset

- **Total evaluation samples**: 2,441

## Retrieval Performance @ 9

| Metric | MM-Embed | LILaC |
|-|-|-|
| Match Exists | 1,215 (0.4977) | **2,138 (0.8759)** |
| Perfect Match | 722 (0.2958) | **1,379 (0.5649)** |
| MRR | – | **0.7324** |

### Notes
- *Match Exists*: At least one gold document appears in top-9 retrieved results
- *Perfect Match*: All required gold documents are retrieved

## LLM Query Performance

| Metric | Top 3 | Top 9 |
|-|-|-|
| Exact Match | 39.86% | **43.34%** |
| F1 Score | 42.16% | **45.82%** |

## Implementation Details

- Retriever: MM-Embed (baseline), LILaC retriever
- LLM: Qwen3-VL
- Retrieval: Top-9 with Beam Size 30
- Evaluation metrics: EM, F1, MRR

> Some implementation details may differ from the original paper due to unavailable components or API constraints.

## Limitations

- Not all hyperparameters from the original paper are publicly available
- Results may vary depending on the LLM and embedding model used
- This implementation prioritizes structural fidelity over performance tuning

## Citation

If you use this code, please cite the original LILaC paper:

```bibtex
@inproceedings{yun2025lilac,
  title     = {LILaC: Late Interacting in Layered Component Graph for Open-domain Multimodal Multihop Retrieval},
  author    = {Yun, Joohyung and Lee, Doyup and Han, Wook-Shin},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2025.emnlp-main.1037}
}
```