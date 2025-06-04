# LSC-Eval  
*A General Framework for Evaluating Methods to Detect Dimensions of Lexical Semantic Change Using LLM-Generated Synthetic Data*

![ACL Findings](https://img.shields.io/badge/ACL%20Findings-Accepted-blueviolet)

---

## 📄 About

**Lexical Semantic Change (LSC)** provides insight into evolving cultural and social dynamics. Yet, the validity of methods for measuring different kinds of LSC remains unestablished due to the absence of historical benchmark datasets.  

To address this gap, we introduce **LSC-Eval**, a novel three-stage general-purpose evaluation framework designed to:
1. **Generate synthetic datasets** that simulate theory-driven LSC using In-Context Learning and a lexical database;
2. **Evaluate the sensitivity** of computational methods to synthetic change; and
3. **Assess method suitability** for detecting change within and across specific dimensions and domains.

LSC-Eval is applied to simulate change along the **Sentiment**, **Intensity**, and **Breadth** (SIB) dimensions—defined in the [SIBling framework](https://github.com/naomibaes/SIBling-framework)—using case studies from psychology. We then evaluate how well selected methods detect these controlled interventions.  

Our findings validate the use of synthetic benchmarks, demonstrate that tailored methods reliably detect changes along SIB dimensions, and reveal that a state-of-the-art LSC model struggles to detect affective aspects of semantic change.  

**LSC-Eval** offers a reusable, extensible tool for benchmarking LSC methods across dimensions and domains, with particular relevance to the social sciences.

---
## 📁 Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `0.0_corpus_preprocessing/` | Scripts for corpus preprocessing (e.g., sentence parsing, filtering for articles with targets) |
| `0.1_descriptives/` | Scripts to compute descriptive statistics |
| `1_sentiment/` | Sentiment-specific scripts and results (Valence index, ABSA Sentiment score) |
| `2_breadth/` | Breadth-specific encoding, manipulation scripts, and results (Breadth score: MPNet and XL-LEXEME) |
| `3_intensity/` | Intensity-specific evaluation scripts (Arousal index) |
| `4_general_LSC/` | Scripts for computing general LSC score (using XL-LEXEME) |
| `figures/` | Plotting scripts and figure outputs (contains some exploratory figures that were not published) |
| `model_comparison/` | Scripts for method/model comparison analyses using synthetic change detection task in bootstrap experimental setup (50 sentences x 100 iterations per injection level) |
| `supplementary_materials/` | Contains top siblings used in replacement strategies for each sampling strategies to give an idea of distribution of injected breadth contexts for each target (bootstrap and five year sampling) -- could not fit in paper |
| `lexeme_utils.py` | Utility functions for lexeme-level manipulation |
| `xlmr_utils.py` | XLM-R embedding computation and encoding scripts (did not end up proceeding with analysis, but good additional model to compare against as XLL was build on it) |
| `requirements.txt` | Required packages for this project (may be incomplete) |
| `README.md` | You are here! |

---

## 🧪 Dimensions of Change

LSC-Eval operationalizes and benchmarks change detection across three semantic dimensions:

| Dimension  | Description |
|------------|-------------|
| **Sentiment**  | Affective polarity: positive ↔ negative |
| **Intensity**  | Affective strength: mild ↔ intense |
| **Breadth**    | Semantic generality: narrow ↔ broad |

Each dimension is injected via targeted synthetic interventions, allowing for fine-grained benchmarking of model sensitivity.

---

## 🚀 Getting Started

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate and install requirements:**
   ```bash
   # Mac/Linux
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate

   pip install -r requirements.txt
   ```

3. **Add your OpenAI API key:**
   ```bash
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

4. **Launch the main notebook:**
   ```bash
   jupyter notebook produce_variations.ipynb
   ```

---

## 🔗 Companion Resources

This repo is part of a broader evaluation ecosystem.

- 📁 **Synthetic Dataset Generation (Psychology domain)**:  
  [Synthetic-LSC Pipeline](https://github.com/naomibaes/Synthetic-LSC_pipeline)  
  Includes full synthetic SIB datasets for 6 psychology targets.

- 📁 **Psychology Corpus (Input Source)**:  
  [Psychology Corpus](https://github.com/naomibaes/psychology_corpus)  
  Sentence-level, year-stamped academic abstracts from psychology articles.

---

## 📬 Contact

For questions, suggestions, or collaboration inquiries, contact:  
**Naomi Baes**  
📧 naomi_baes@hotmail.com  
🌐 [naomibaes.github.io](https://naomibaes.github.io)

---

## 🙏 Acknowledgements

Special thanks to **Raphael Merx** for foundational contributions to the synthetic generation pipeline, ABSA sentiment classification integration, and XL-LEXEME embeddings.

This work was developed in collaboration with  
**Haim Dubossarsky, Ekaterina Vylomova, and Nick Haslam.**
