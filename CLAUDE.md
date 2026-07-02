# Project Context for Claude Code

## What this repo is
Lecture notes for an ML in Computational Physics course (not exam-specific — general course notes), published as a website at https://hbarquanta.github.io/MLinCP-Notes/ via MkDocs Material.

## Repo layout
- `docs/` — the source of truth, one markdown file per chapter (`chapter-01-...md` – `chapter-10-...md`) plus `docs/index.md` (landing page) and `docs/javascripts/katex.js` (math rendering config). Tracked, pushed.
- `mkdocs.yml` — site config: nav, theme, math (KaTeX via `pymdownx.arithmatex`), search.
- `.github/workflows/deploy.yml` — builds and deploys to the `gh-pages` branch on every push to `main` via `mkdocs gh-deploy`.
- `lectures/` — source lecture PDF slide decks, renamed `L01_...pdf`–`L10_...pdf` for consistency. Gitignored, local-only, never pushed (copyright).
- `site/` — MkDocs build output. Gitignored — never commit it; the GitHub Action regenerates it on `gh-pages`.
- Only `docs/`, `mkdocs.yml`, `.github/`, `README.md`, and repo config files should be pushed to the public repo.

## User preferences
- Concise and direct — no fluff
- Block math ($$...$$) always, never inline for standalone equations
- Q&A as plain text, not widgets
- Q&A: few consolidated questions per section, follow the order of the notes, detailed answers with equations

## Markdown authoring gotchas (learned the hard way)
- **Always leave a blank line before and after a `$$...$$` block equation.** Without it, the equation gets merged into the preceding paragraph and MkDocs' math extension (`pymdownx.arithmatex`) fails to recognize it as display math — it renders as literal text instead. (Docsify's old client-side hack tolerated this; MkDocs' proper block-level parser does not.)
- **Escape literal `|` characters inside math that appears in a markdown table cell** (e.g. `\|A\|` for cardinality, not `|A|`) — an unescaped `|` is read as a table column separator and splits the cell, breaking both the table and the equation.

## Notes structure
10 chapters, each its own page under `docs/`:
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
- Occasional interactive demos (sliders/knobs driving a live plot) are in scope later — prefer an embedded Observable notebook (`<iframe>`) or a small Plotly.js snippet per-widget, not a framework migration.
