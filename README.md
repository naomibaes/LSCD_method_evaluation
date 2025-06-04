# LSC-Eval  
*A General Framework for Evaluating Methods to Detect Dimensions of Lexical Semantic Change Using LLM-Generated Synthetic Data*

📄 **Citation**:  
Baes, N., Merx, R., Haslam, N., Vylomova, E., & Dubossarsky, H. (2025).  
*LSC-Eval: A General Framework to Evaluate Methods for Assessing Dimensions of Lexical Semantic Change Using LLM-Generated Synthetic Data.*  
_arXiv preprint_, arXiv:2503.08042. [View on arXiv](https://arxiv.org/abs/2503.08042)

![ACL Findings](https://img.shields.io/badge/ACL%20Findings-Accepted-blueviolet)

---

## 📄 About

**Lexical Semantic Change (LSC)** provides insight into evolving cultural and social dynamics. Yet, the validity of methods for measuring different kinds of LSC remains unestablished due to the absence of historical benchmark datasets.

To address this gap, we introduce **LSC-Eval**, a general-purpose three-stage evaluation framework designed to benchmark computational LSC methods under controlled, interpretable conditions:

1. **Stage 1 – Synthetic Dataset Generation:**  
   Create synthetic corpora that simulate theory-driven semantic changes using In-Context Learning (ICL) with LLMs and lexical resources. This includes targeted manipulations along three semantic dimensions—**Sentiment**, **Intensity**, and **Breadth**—as formalized in the [SIBling framework](https://aclanthology.org/2024.acl-long.76/).

2. **Stage 2 – Method Evaluation:**  
   Quantify each method’s ability to detect dimension-specific change by computing semantic scores (e.g., Valence, Arousal, Breadth) across injection levels and time bins. We test models such as ABSA classifiers and contextual embedding distances (MPNet, XL-LEXEME).

3. **Stage 3 – Sensitivity & Suitability Analysis:**  
   Systematically compare methods under a bootstrapped experimental setup to assess their **sensitivity** to controlled interventions and **suitability** for detecting LSC in specific semantic dimensions and domains.

We apply LSC-Eval to six psychology-related concepts and evaluate method performance in detecting changes introduced via synthetic interventions.  

Our findings support the validity of synthetic benchmarks, show that targeted methods reliably track changes along SIB dimensions, and reveal that even state-of-the-art LSC models underperform on affective dimensions (Sentiment, Intensity).

**LSC-Eval** thus offers a general-purpose framework for evaluating LSC methods in a theory-driven, dimension- and domain-specific way, with particular value for the social sciences.

---

## 📁 Repository Structure

| Folder/File | Description |
|-------------|-------------|
| `0.0_corpus_preprocessing/` | Scripts for corpus preprocessing (e.g., sentence parsing, filtering for target terms) |
| `0.1_descriptives/` | Scripts to compute descriptive statistics |
| `1_sentiment/` | Sentiment-specific scripts and results (Valence index, ABSA Sentiment score) |
| `2_breadth/` | Breadth-specific encoding, manipulation scripts, and results (Breadth score via MPNet and XL-LEXEME) |
| `3_intensity/` | Intensity-specific evaluation scripts (Arousal index) |
| `4_general_LSC/` | Scripts for computing the general LSC score (using XL-LEXEME) |
| `figures/` | Plotting scripts and figure outputs (some exploratory figures not included in the paper) |
| `model_comparison/` | Scripts for method/model comparison using the synthetic change detection task (bootstrap setup: 50 sentences × 100 iterations per injection level) |
| `supplementary_materials/` | Contains top sibling terms used in replacement strategies, plus distributional metadata from breadth injection (bootstrap and 5-year sampling) |
| `lexeme_utils.py` | Utility functions for lexeme-level manipulation |
| `xlmr_utils.py` | XLM-R embedding computation and encoding scripts (analysis not included in the paper, but useful for comparing against XL-LEXEME) |
| `requirements.txt` | Required packages (may need to be expanded based on usage) |
| `README.md` | You are here! |

---

## 🟢 🔴 🔵 Dimensions of Change

LSC-Eval operationalizes and benchmarks change detection across three semantic dimensions, as defined in the **SIBling framework** ([Baes et al., 2024](https://aclanthology.org/2024.acl-long.76/)):

| Dimension   | Definition | Example of Rising | Example of Falling |
|-------------|------------|--------------------|---------------------|
| 🟢 **Sentiment** | Refers to the degree to which a word’s meaning acquires more positive (*elevation*, *amelioration*) or negative (*degeneration*, *pejoration*) connotations. | *craftsman* (once manual labor, now implies high skill); *geek* (derogatory → enthusiast) | *retarded* (neutral clinical → pejorative); *awful* (once awe-inspiring → very bad) |
| 🔴 **Intensity** | Refers to the degree to which a word’s meaning becomes more (*hyperbole*) or less (*meiosis*) emotionally or referentially intense—e.g., stronger, more potent, or higher-arousal in meaning. | *cool* (from temperature to strong approval); *hilarious* (from cheerful to extreme laughter) | *love* (expanded to mild liking); *trauma* (from physical injury to mild adversity) |
| 🔵 **Breadth** | Refers to the degree to which a word’s semantic range expands (*widening*, *generalization*) or contracts (*narrowing*, *specialization*), such as shifts in category, scope, or contextual usage. | *cloud* (meteorology → data storage); *partner* (business → romantic/domestic) | *doctor* (broad → mostly medical); *meat* (any food → animal flesh) |

Each dimension is evaluated independently using **targeted synthetic interventions** applied to natural corpus sentences across 5-year intervals. This enables fine-grained benchmarking of whether models can detect subtle and dimension-specific semantic shifts over time.


## 🔗 Companion Resources

This repository is part of a broader evaluation ecosystem. While it implements the general three-stage **LSC-Eval** framework using examples from psychology, the framework itself is **domain-agnostic** and can be extended to other dimensions—including non-LSC semantic dimensions—and applied across domains.

Current related resources include:

- 📁 **Synthetic-LSC Pipeline (Psychology domain)**  
  [Synthetic-LSC Pipeline](https://github.com/naomibaes/Synthetic-LSC_pipeline)  
  Contains synthetic datasets simulating Sentiment, Intensity, and Breadth (SIB) for six psychology-related target terms.

- 📁 **Psychology Corpus (Input Source)**  
  [Psychology Corpus](https://github.com/naomibaes/psychology_corpus)  
  A year-stamped corpus of article abstracts from academic psychology journals.

---

## 🚀 Getting Started (incomplete)

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

Special thanks to **Raphael Merx** for foundational contributions to the synthetic generation pipeline, the use of ABSA classification (to classify sentiment using the word-in-context approach), and the use of the word transformer XL-LEXEME for embedding-based evaluation.

This work was developed in collaboration with, and under the supervision of:  
**Haim Dubossarsky, Ekaterina Vylomova, and Nick Haslam.**
