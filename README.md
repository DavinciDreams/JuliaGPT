---
language: en
license: mit
library_name: custom
tags:
  - gpt
  - character-level
  - transformer
  - from-scratch
  - ancient-scripts
  - classical-texts
datasets:
  - custom
pipeline_tag: text-generation
---

# JuliaGPT

An optimized character-level GPT in Julia for training on ancient scripts and classical texts. Evolution of [MicroJulia](https://github.com/DavinciDreams/micro-julia).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavinciDreams/JuliaGPT/blob/main/juliagpt.ipynb)

## Roadmap

Starting from MicroJulia's minimal scalar-autograd GPT, optimizing toward:

- Array-based autograd for 100-1000x speedup
- Multi-layer transformers with GELU activations
- Learnable RMSNorm, gradient clipping, cosine LR schedule
- Ancient script support (Greek, Latin, Cuneiform, etc.)
- Flexible vocabulary configuration per script
- Batched training and proper attention masking

## Current Architecture

- Custom autograd engine in pure Julia
- Transformer with multi-head attention
- Character-level tokenization
- Adam optimizer with LR decay
- W&B logging + HuggingFace Hub integration

## Quick Start

1. Click "Open in Colab" above
2. Add Colab secrets: `HF_TOKEN`, `WANDB_KEY`, `HF_REPO`
3. Run Python login cell, install Julia, switch to Julia 1.10
4. Run all cells

## Related

- [micro-julia](https://github.com/DavinciDreams/micro-julia) - Original minimal implementation
- [text-pipeline](https://github.com/DavinciDreams/text-pipeline) - Text processing pipeline for training data
