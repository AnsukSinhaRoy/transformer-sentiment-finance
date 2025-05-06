# Transformer from Scratch for Financial Sentiment Analysis

This project implements a Transformer architecture **from scratch (excluding word embeddings)** for performing sentiment analysis on financial texts such as market news, stock discussions, and earnings reports.

## 🔍 Objective

- Implement a configurable Transformer model for analyzing sentiment in financial data.
- Focus on low-level implementation: Multi-head attention, positional encoding, feed-forward layers, residuals, layer normalization, etc.
- Use **pretrained 1024-dimensional word embeddings** (e.g., GloVe, fastText, FinBERT). Word embedding creation is out of scope.

## 🏗️ Features

- Manual control over Transformer architecture:
  - Number of layers
  - Number of attention heads
  - Custom `Wq`, `Wk`, `Wv` dimensions
  - Feedforward network size
  - Dropout rates and activation functions
- Sentiment classification head
- Clean modular design for easy experimentation

## 📁 Project Structure

```
transformer-sentiment-finance/
├── data/                    # Financial text datasets
├── embeddings/              # Pretrained 1024-dim word embeddings (input only)
├── models/
│   ├── __init__.py
│   ├── transformer.py       # Core Transformer implementation
│   ├── attention.py         # Multi-head self-attention from scratch
│   ├── encoder.py           # Transformer encoder block
│   └── positional_encoding.py # Positional encoding implementation
├── utils/
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── metrics.py           # Accuracy, loss, confusion matrix
│   └── config.py            # Hyperparameter configuration
├── train.py                 # Training loop
├── evaluate.py              # Model evaluation and analysis
├── config.json              # Model and training hyperparameters
└── README.md                # Project overview
```

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
```

## 📊 Dataset (Planned)

- Financial PhraseBank
- FiQA Sentiment Dataset
- Market sentiment news datasets

## 🚀 Training

```bash
python train.py --config config.json
```

## 🧠 You Decide

You are responsible for:
- Selecting architecture dimensions (e.g., attention sizes, number of heads)
- Deciding model depth (layers)
- Choosing activation functions, optimizer, and learning schedule

## 📌 TODO

- [ ] Build multi-head attention
- [ ] Add positional encoding
- [ ] Implement encoder block
- [ ] Build complete Transformer encoder
- [ ] Create training and evaluation loop
- [ ] Integrate with financial datasets
- [ ] Tune architecture and analyze results

---
