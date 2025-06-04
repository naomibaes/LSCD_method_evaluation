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

LSC-Eval operationalizes and benchmarks change detection across three semantic dimensions, as defined in the **SIBling framework** ([Baes et al., 2024](https://aclanthology.org/2024.acl-long.76/)):

| Dimension   | Definition |
|-------------|------------|
| **Sentiment** | Refers to the degree to which a word’s meaning acquires more positive (*elevation*, *amelioration*) or negative (*degeneration*, *pejoration*) connotations. |
| **Intensity** | Refers to the degree to which a word’s meaning becomes more (*hyperbole*) or less (*meiosis*) emotionally or referentially intense—e.g., stronger, potent, or higher-arousal in meaning. |
| **Breadth** | Refers to the degree to which a word’s semantic range expands (*widening*, *generalization*) or contracts (*narrowing*, *specialization*). This can involve shifts in category, scope, or usage context. |

Each dimension is evaluated independently using **targeted synthetic interventions** applied to natural corpus sentences across 5-year intervals. This enables fine-grained benchmarking of whether models can detect subtle and dimension-specific semantic shifts over time.

---
## 🔗 Companion Resources

This repository is part of a broader evaluation ecosystem. While it implements the general three-stage **LSC-Eval** framework using examples from psychology, the framework itself is **domain-agnostic** and can be extended to other dimensions of change—including non-LSC dimensions—and applied across domains.

Below are its current applications:

- 📁 **Synthetic-LSC Pipeline (Psychology domain)**  
  [Synthetic-LSC Pipeline](https://github.com/naomibaes/Synthetic-LSC_pipeline)  
  Contains synthetic datasets simulating Sentiment, Intensity, and Breadth (SIB) for six psychology-related target concepts.

- 📁 **Psychology Corpus (Input Source)**  
  [Psychology Corpus](https://github.com/naomibaes/psychology_corpus)  
  A year-partitioned corpus of article abstracts from academic psychology journals.

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

## 📬 Contact

For questions, suggestions, or collaboration inquiries, contact:  
**Naomi Baes**  
📧 naomi_baes@hotmail.com  
🌐 [naomibaes.github.io](https://naomibaes.github.io)

---

## 🙏 Acknowledgements

Special thanks to **Raphael Merx** for foundational contributions to the synthetic generation pipeline, ABSA sentiment classification idea (to classify sentiment using the word-in-context approach), and use XL-LEXEME to get word transformer embeddings.

This work was developed in collaboration with, and under the supervision of:  
**Haim Dubossarsky, Ekaterina Vylomova, and Nick Haslam.**
