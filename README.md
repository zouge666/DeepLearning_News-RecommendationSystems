# 📰 Deep Learning News Recommendation Systems

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-EE4C2C.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%3E%3D2.12.0-FF6F00.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

With the rapid growth of online content, personalized news recommendation systems are critical for helping users navigate vast information spaces. However, traditional systems over-optimize for Click-Through Rates (CTR), often creating echo chambers and reducing content diversity.

This project tackles these challenges by building a robust, state-of-the-art news recommendation system that balances **accuracy** and **fairness**. Leveraging the Ekstra Bladet RecSys 2024 dataset, we utilize advanced deep learning paradigms—such as Transformers and Attention mechanisms—to deeply model both user behaviors and article semantics.

## Key Features & Architecture

- **Neural News Recommendation with Multi-Head Self-Attention (NRMS):** Implements a dual-encoder architecture (News Encoder & User Encoder) with a Click Predictor to capture nuanced user interests.
- **Advanced Representation Learning:** Combines user behavioral data (browsing history, session dynamics) with textual content using self-attention networks.
- **Beyond CTR Optimization:** Explores fairness-aware recommendation strategies to prevent topic saturation and enhance content diversity.
- **Experimental & Future Work:** - Processing long sequences using **Transformer-XL / Longformer**.
  - Incorporating **Graph Neural Networks (GNNs)** for complex user-news relationship modeling.
  - Integrating multimodal data and sentiment/topic analysis.
  - Utilizing **Deep Reinforcement Learning (DRL)** for dynamic recommendation strategies.

## 📌 Installation & Setup

The project uses `pyproject.toml` for standard dependency management. It requires **Python 3.10 or 3.11**.

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/your-username/DeepLearning_News-RecommendationSystems.git](https://github.com/your-username/DeepLearning_News-RecommendationSystems.git)
   cd DeepLearning_News-RecommendationSystems
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```
   _(Optional)_ To install dependencies for Jupyter Notebooks or running tests:
   ```bash
   pip install -e .[notebooks]
   pip install -e .[tests]
   ```

## Dataset Preparation

This project uses the official RecSys 2024 Dataset from Ekstra Bladet.

1. Download the dataset from https://recsys.eb.dk.
2. Extract the downloaded files.
3. Place the data into the project structure (e.g., inside a `data/raw/` directory) so that the preprocessing scripts can locate the interaction logs and article metadata.

## Milestones & Quick Start

### 1. Data Processing

Preprocess the raw Ekstra Bladet dataset, clean the behaviors/histories, and generate textual embeddings.

```bash
# Example script (update path based on your exact script structure)
python scripts/preprocess_data.py --data_dir ./data/raw --out_dir ./data/processed
```

### 2. Model Training (NRMS)

Train the deep learning model. The architecture automatically instantiates the News Encoder, User Encoder, and Click Predictor.

```bash
python scripts/train_nrms.py --config configs/nrms_config.yaml
```

### 3. Evaluation

Evaluate the model against the hidden test set. The primary metrics used to benchmark performance are AUC (Area Under the ROC Curve) and nDCG (Normalized Discounted Cumulative Gain).

```bash
python scripts/evaluate.py --model_weights ./checkpoints/best_model.pt
```

## References & Core Literature

This system is heavily inspired by and builds upon the following seminal research:

1. Wu, C. et al. "Neural News Recommendation with Multi-Head Self-Attention." EMNLP-IJCNLP (2019).
2. Huang, J. et al. "Adapted transformer network for news recommendation."
3. Gao, Y. et al. "Content Filtering Enriched GNN Framework for News Recommendation."
4. Zheng, G. et al. "DRN: A Deep Reinforcement Learning Framework for News Recommendation." WWW (2018).

---
