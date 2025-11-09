
# Legal Clause Similarity Detection
This repository implements baseline deep-learning models to detect semantic similarity between legal clauses without using pretrained transformers. It includes complete data loading, preprocessing, two model architectures (BiLSTM and ESIM), training pipeline, comprehensive evaluation metrics, and visualizations.

## 🚀 Quick Start

**Prerequisites:** Python 3.8+ and Git. On Windows, use PowerShell.

### 1. Clone the repository

```powershell
git clone https://github.com/tahahasan01/deep-learning-legal-clause-similarity.git
cd deep-learning-legal-clause-similarity
```

### 2. Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the complete pipeline

Open and run the Jupyter notebook which contains the complete end-to-end pipeline:

```powershell
jupyter notebook Legal_Clause_Similarity_A2.ipynb
```

**Note:** The notebook is self-contained and includes all necessary code. Simply run all cells from top to bottom to:
- Load and preprocess data
- Train both BiLSTM and ESIM models
- Evaluate on test set
- Generate comprehensive visualizations

The training takes approximately 5-10 minutes on GPU (or 15-20 minutes on CPU) with early stopping enabled.

## 📁 Project Structure

```
deep-learning-legal-clause-similarity/
├── Legal_Clause_Similarity_A2.ipynb   # Main notebook with complete pipeline
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── PROJECT_SUMMARY.md                 # Detailed project documentation
├── archive (1)/                       # Dataset (300+ CSV files with legal clauses)
│   ├── access.csv
│   ├── arbitration.csv
│   ├── confidentiality.csv
│   └── ... (300+ category files)
└── artifacts/                         # Generated outputs (created automatically)
    ├── best_bilstm.pth               # Trained BiLSTM model checkpoint
    ├── best_esim.pth                 # Trained ESIM model checkpoint
    ├── metrics_bilstm.json           # BiLSTM evaluation metrics
    └── metrics_esim.json             # ESIM evaluation metrics
```

**Note:** The notebook is self-contained with all functions defined inline. Supporting Python files (`data_loader.py`, `models.py`, `trainer.py`, etc.) exist for reference but are not required to run the notebook.

## 📊 Dataset

The `archive (1)` folder contains 300+ CSV files, where each file corresponds to a specific legal clause category (e.g., `access.csv`, `arbitration.csv`, `confidentiality.csv`, `indemnification.csv`, etc.).

**Data Processing:**
- **Positive pairs:** Two clauses from the same category (semantically similar)
- **Negative pairs:** Two clauses from different categories (semantically dissimilar)
- **Deduplication:** Removes duplicate clauses to prevent data leakage
- **Splits:** 70% train, 15% validation, 15% test (stratified by category)
- **Data integrity:** Zero clause overlap between splits verified automatically

**Pair Generation:**
- Training: Up to 200 positive pairs per category + balanced negative pairs
- Validation/Test: Up to 50 positive pairs per category + balanced negative pairs
- Total pairs: ~40,000 training, ~10,000 validation, ~10,000 test

## 🤖 Models Implemented

### 1. BiLSTM Similarity Model
- **Architecture:** Bidirectional LSTM encoder with mean pooling
- **Similarity Features:** Concatenation, absolute difference, element-wise multiplication
- **Classifier:** 2-layer fully connected network with ReLU and dropout
- **Parameters:** ~500K trainable parameters

### 2. ESIM (Enhanced Sequential Inference Model) Lite
- **Architecture:** LSTM encoder with soft attention alignment
- **Key Features:** 
  - Cross-attention between clause pairs
  - Composition layer for enhanced representations
  - Max and mean pooling aggregation
- **Parameters:** ~600K trainable parameters

**Training Configuration:**
- Optimizer: Adam (lr=1e-3, weight_decay=1e-5)
- Loss: Cross-entropy
- Batch size: 64
- Max epochs: 20
- Early stopping: 3 epochs without improvement
- Device: GPU (CUDA) if available, else CPU

## 📈 Results & Evaluation

Both models achieve excellent performance on the legal clause similarity task:

| Model  | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|--------|----------|-----------|--------|----------|---------|--------|
| BiLSTM | 99.91%   | 99.90%    | 99.98% | 99.94%   | 99.90%  | 99.96% |
| ESIM   | 99.87%   | 99.85%    | 99.99% | 99.92%   | 99.84%  | 99.91% |

**Winner:** BiLSTM (marginally better F1-score: 99.94% vs 99.92%)

### Visualizations Generated

The notebook automatically generates the following visualizations (displayed inline):

1. **Training Curves:** Loss and accuracy over epochs for both models
2. **Confusion Matrices:** Classification performance breakdown
3. **ROC Curves:** Receiver Operating Characteristic with AUC scores
4. **Precision-Recall Curves:** Performance across different thresholds
5. **Side-by-Side Comparison:** Bar charts comparing all metrics
6. **Training History:** Validation accuracy convergence comparison

All visualizations appear directly in the notebook output - no external image files needed!

## 🔬 Evaluation Metrics

The notebook computes comprehensive metrics:

- **Accuracy:** Overall classification correctness
- **Precision:** Positive predictive value
- **Recall:** Sensitivity/true positive rate
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve (discrimination ability)
- **PR-AUC:** Area under precision-recall curve (performance with imbalanced data)
- **Confusion Matrix:** True/false positives and negatives
- **Training History:** Loss and accuracy tracked per epoch

## ✅ Reproducibility & Data Integrity

**Reproducibility Measures:**
- Fixed random seeds throughout the pipeline
- Deterministic data splits (70/15/15 train/val/test)
- Checkpoint saving for best models
- Metrics saved to JSON for reference

**Data Integrity Checks:**
The notebook includes automatic verification:
- ✓ Zero clause overlap between train/val/test splits
- ✓ Vocabulary built exclusively from training data
- ✓ No data leakage across splits
- ✓ Assertions fail if any integrity issues detected

## 💡 Key Features

- **Self-contained notebook:** All code in one place, easy to understand and modify
- **Inline visualizations:** Charts display directly in notebook (no PNG files needed)
- **Automatic checkpointing:** Best models saved based on validation F1-score
- **Early stopping:** Prevents overfitting, stops when validation metrics plateau
- **GPU acceleration:** Automatic CUDA detection and usage if available
- **Comprehensive logging:** Progress printed for every major step
- **Data leakage prevention:** Strict split enforcement with automated checks

## 🛠️ Tips & Troubleshooting

**Memory issues?**
- Reduce `batch_size` in the config cell (currently 64)
- Reduce `max_pos_per_cat` and `max_neg_pairs` in pair generation
- Use a smaller `max_vocab` size (currently 60,000)

**Training too slow?**
- Verify GPU is being used: Check for "CUDA available: True" in output
- Increase `batch_size` if you have more GPU memory
- Reduce `epochs` or use more aggressive `early_stop`

**Models not training (loading from checkpoint)?**
- Delete `.pth` files in `artifacts/` folder to force retraining
- Or modify the training cell to skip checkpoint loading

**Want to retrain from scratch?**
- Delete or rename the `artifacts/` folder
- Re-run the notebook from the training cell onwards

## 🚀 Suggested Extensions

Potential improvements for future work:

- **Pretrained embeddings:** Add GloVe or FastText for better word representations
- **Transformer models:** Fine-tune BERT/RoBERTa for state-of-the-art performance
- **Data augmentation:** Paraphrasing, back-translation, or synonym replacement
- **Ensemble methods:** Combine BiLSTM and ESIM predictions
- **Hyperparameter tuning:** Grid search or Bayesian optimization
- **Cross-validation:** K-fold CV for more robust evaluation
- **Explainability:** Add attention visualization or LIME explanations
- **API deployment:** Flask/FastAPI endpoint for inference

## 📦 Dependencies

Key libraries (see `requirements.txt` for full list):
- `torch>=2.0.0` - Deep learning framework
- `scikit-learn>=1.3.0` - Metrics and evaluation
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualizations
- `jupyter>=1.0.0` - Notebook interface

## 📞 Contact

**Student:** Syed Taha Hasan  
**FastID:** i211767  
**Course:** CS495 - Deep Learning

For questions about reproducing experiments or implementation details, please:
1. Check the notebook (`Legal_Clause_Similarity_A2.ipynb`) first - it contains detailed comments
2. Review `PROJECT_SUMMARY.md` for architectural details
3. Open an issue in the repository

## 📄 License

This repository is for educational purposes for the CS495 Deep Learning course at FAST-NUCES.

---

**Note:** This project demonstrates fundamental deep learning techniques for NLP without relying on pretrained transformers, showcasing understanding of core concepts like sequence modeling, attention mechanisms, and similarity learning.


