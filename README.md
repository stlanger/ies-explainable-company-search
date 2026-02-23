# IES Explainable Company Search


Innovation Explainable Company Search  is an interpretable retrieval-augmented generation (RAG) framework designed to support small and medium-sized enterprises (SMEs) in identifying regional partners through transparent and trustworthy AI-assisted search.

The system combines structured company metadata with unstructured website content and applies a context-aware chunking and retrieval strategy to generate relevance scores, summaries, and explanations. This Project is explicitly designed to operate in computationally limited settings, lowering the barrier to entry for digitally underserved organizations.

## Key Features

- Explainable RAG pipeline
Retrieval-based context selection combined with LLM-powered summarization and explanation generation.

- Context-window chunking strategy
A lightweight mechanism to improve explanation coherence and traceability.

- Interpretable relevance scoring
Explanation scores designed to support human decision-making.

- Multilingual company datasets (UK & Germany)
Harmonized structured and unstructured business data for benchmarking and research. The dataset has been published with the DOi [`10.24352/UB.OVGU-2026-022`](https://commons.datacite.org/doi.org/10.24352/ub.ovgu-2026-022) and ist also available on [Github](https://github.com/stlanger/ies-companies-dataset).

- Resource-efficient design
Suitable for deployment on modest hardware infrastructures.

## Research Contributions

This repository accompanies our research work on explainable AI-driven partner discovery. We provide:
- The full IES implementation
- Dataset construction pipeline
- Evaluation scripts and baseline comparisons
- Reproducible experimental setup

## Intended Use

IES is intended for:
- Research on explainable retrieval-augmented generation
- SME-focused AI systems
- Trustworthy AI and human-centered evaluation
- Regional innovation and partner discovery analysis

## License

See the [LICENSE](./LICENSE) file for details.