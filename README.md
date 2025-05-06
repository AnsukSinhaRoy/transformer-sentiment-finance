# Transformer from Scratch for Financial Sentiment Analysis

This project implements a Transformer architecture **from scratch (excluding word embeddings)** for performing sentiment analysis on financial texts like market news, stock discussions, and earnings reports.

## 🔍 Objective

- Implement a configurable Transformer model for analyzing sentiment in financial data.
- Focus on raw model implementation (Multi-head attention, position encodings, feed-forward layers, layer norms, etc.).
- Use pretrained 1024-dimensional word embeddings (e.g., GloVe, fastText, or FinBERT outputs).

## 🏗️ Features

- Manual control of model architecture:
  - Number of transformer layers
  - Number of attention heads
  - Custom `Wq`, `Wk`, `Wv` dimensions
  - Feedforward network size
  - Dropout and activation functions
- Sentiment classification head on top of the Transformer encoder.
- Designed for flexibility in analyzing stock-related sentiment.

## 📁 Project Structure

├── data/ # Financial text datasets
├── embeddings/ # Pre-trained 1024-dim embeddings (not trained here)
├── models/
│ └── transformer.py # Core transformer implementation
├── train.py # Training loop and logging
├── evaluate.py # Evaluation metrics and visualizations
├── utils/ # Data loaders, tokenizer adapters
├── config.json # Model hyperparameter config
└── README.md


## 🔧 Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- pandas
- tqdm

Install with:

```bash
pip install -r requirements.txt
