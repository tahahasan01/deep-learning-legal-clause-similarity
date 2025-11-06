# Legal Clause Similarity Detection - Assignment 2

**Course:** Deep Learning (CS425)  
**Student:** Syed Taha Hasan  
**FastID:** i211767

## Project Overview

This project implements NLP models to identify semantic similarity between legal clauses using baseline architectures without pre-trained transformers. Two models are implemented:

1. **BiLSTM Similarity Model** - Bidirectional LSTM with mean pooling and similarity features
2. **Attention-based Encoder Model** - Transformer encoder with self-attention and cross-attention mechanisms

## Project Structure

```
.
├── data_loader.py              # Data loading and pair generation
├── text_preprocessor.py        # Text preprocessing and tokenization
├── models.py                   # Model architectures (BiLSTM and Attention)
├── trainer.py                  # Training pipeline and utilities
├── evaluator.py               # Evaluation metrics and qualitative analysis
├── visualization.py            # Plotting and visualization utilities
├── Legal_Clause_Similarity_A2.ipynb  # Main notebook
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure the dataset folder `archive (1)` is in the project root directory.

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook Legal_Clause_Similarity_A2.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the legal clause dataset
   - Train both baseline models
   - Evaluate and compare model performance
   - Generate training graphs and qualitative results

## Model Architectures

### BiLSTM Model
- **Embedding Layer:** Word embeddings (128-dim)
- **BiLSTM:** 2 layers, 256 hidden units (bidirectional)
- **Similarity Features:** Concatenation, difference, and element-wise multiplication
- **Classifier:** Fully connected layers with dropout

### Attention-based Encoder Model
- **Embedding Layer:** Word embeddings with positional encoding (128-dim)
- **Transformer Encoder:** 2 layers, 8 attention heads
- **Cross-Attention:** Between the two clause representations
- **Pooling:** Mean and max pooling
- **Classifier:** Fully connected layers with dropout

## Evaluation Metrics

The models are evaluated using:
- **Accuracy:** Overall classification accuracy
- **Precision:** Precision for similar clause pairs
- **Recall:** Recall for similar clause pairs
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the ROC curve
- **PR-AUC:** Area under the Precision-Recall curve

## Dataset

The dataset consists of legal clauses organized by category. Each CSV file contains clauses of a specific legal category (e.g., `acceleration.csv`, `access-to-information.csv`).

- **Similar pairs:** Clauses from the same category
- **Dissimilar pairs:** Clauses from different categories

## Key Features

- ✅ Modular, object-oriented implementation
- ✅ Comprehensive evaluation metrics
- ✅ Training visualization (loss and accuracy curves)
- ✅ Model comparison and analysis
- ✅ Qualitative results (correct/incorrect examples)
- ✅ Early stopping and learning rate scheduling
- ✅ Reproducible results (fixed random seeds)

## Results

The notebook generates:
1. Training history graphs (loss and accuracy over epochs)
2. Model comparison charts
3. Performance metrics table
4. Qualitative examples of correct and incorrect predictions

## Notes

- Models are trained from scratch (no pre-trained embeddings)
- Early stopping is used to prevent overfitting
- Data is split into 70% train, 15% validation, 15% test
- Stratified sampling ensures balanced splits

## License

This project is for educational purposes as part of CS425 Deep Learning course.

