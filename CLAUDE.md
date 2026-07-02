# Project Context for Claude Code

## What this repo is
Lecture notes for an ML in Computational Physics course (not exam-specific — general course notes). The main file is `Notes.md` (~2115 lines, 10 chapters). The goal is to turn this into a GitHub Pages site.

## Repo layout
- `Notes.md` — the public-facing study notes (tracked, pushed).
- `lectures/` — source lecture PDF slide decks, renamed `L01_...pdf`–`L10_...pdf` for consistency. Gitignored, local-only, never pushed (copyright).
- Only `Notes.md`, `README.md`, and repo config files should be pushed to the public repo.

## User preferences
- Concise and direct — no fluff
- Block math ($$...$$) always, never inline for standalone equations
- Q&A as plain text, not widgets
- Q&A: few consolidated questions per section, follow the order of the notes, detailed answers with equations

## Notes structure
10 chapters with a Table of Contents at the top:
- Chapter 1: Core ML Concepts
- Chapter 2: Descriptors & Featurization
- Chapter 3: Dimensionality Reduction & Clustering
- Chapter 4: Regression & Uncertainty Quantification
- Chapter 5: Neural Networks, Trees & Regularization
- Chapter 6: Graph Neural Networks
- Chapter 7: MLIPs I (Behler-Parrinello, GAP, ACE, SchNet, NequIP, MACE)
- Chapter 8: MLIPs II (production quality, active learning, foundation models, long-range)
- Chapter 9: Generative Models (GANs, VAEs, Autoregressive, LLMs/Transformers, NFs, Diffusion, Flow Matching)
- Chapter 10: Rare Events (bias methods, TPS, NF-for-TPS via Falkner et al. 2023, committor, AIMMD)

## Pending tasks
- Fix numbering gaps: Chapter 8 jumps 8.4 → 8.6; Chapter 9 sections have no numbers; Chapter 7 has unnumbered subsections (GAP, ACE, "Epochs of MLIP Development") that should be 7.7, 7.8, etc.
- Add figures, diagrams, and flowcharts
- Language pass: more natural prose, no em-dashes
- Set up GitHub Pages rendering (MathJax/KaTeX for math)
