# LSC-Eval: A General Framework for Evaluating Methods to Detect Dimensions of Lexical Semantic Change Using LLM-Generated Synthetic Data

## Getting Started

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the environment and install dependencies:
   ```bash
   # On Mac/Linux:
   source .venv/bin/activate

   # On Windows:
   .venv\Scripts\activate

   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```

4. Launch and run the notebooks (e.g., below):
   ```bash
   jupyter notebook produce_variations.ipynb
   ```

## 📄 About

This repository contains code for generating synthetic sentence variations to simulate Lexical Semantic Change (LSC) along three dimensions: **Sentiment**, **Intensity**, and **Breadth** (SIB), as defined in the [SIBling framework](https://github.com/naomibaes/SIBling-framework).

The pipeline uses in-context learning with large language models (LLMs), combined with sentiment classification (ABSA), embedding-based evaluation, and lexical resources.

## 📂 Synthetic Datasets

See [this companion repository](#) for:
- Synthetic datasets simulating SIB dimensions for six psychology-related target concepts
- Scripts to run evaluation tasks using these datasets

## 📬 Contact

For questions or collaboration inquiries, please contact:  
**Naomi Baes**  
📧 naomi_baes@hotmail.com

## 🙏 Acknowledgements

Special thanks to **Raphael Merx** for his contributions to the LLM-based generation pipeline, ABSA classification integration, and support with XL-LEXEME embeddings.
