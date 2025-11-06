"""
Evaluation Module
Computes various evaluation metrics for model performance
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluates model performance using multiple metrics"""
    
    def __init__(self, model: torch.nn.Module, device: str = None):
        """
        Initialize evaluator
        
        Args:
            model: PyTorch model
            device: Device to use
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and true labels
        
        Args:
            dataloader: Data loader
            
        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                text1 = batch['text1'].to(self.device)
                text2 = batch['text2'].to(self.device)
                labels = batch['label'].squeeze().cpu().numpy()
                
                # Forward pass
                outputs = self.model(text1, text2)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels)
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probs)
    
    def compute_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Compute all evaluation metrics
        
        Args:
            dataloader: Data loader
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels, probs = self.predict(dataloader)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # AUC metrics
        try:
            roc_auc = roc_auc_score(labels, probs)
        except ValueError:
            roc_auc = 0.0
        
        try:
            pr_auc = average_precision_score(labels, probs)
        except ValueError:
            pr_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], model_name: str = "Model"):
        """
        Print metrics in a formatted way
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        print(f"\n{'='*60}")
        print(f"{model_name} - Evaluation Metrics")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"{'='*60}\n")
    
    def plot_confusion_matrix(self, dataloader: DataLoader, model_name: str = "Model"):
        """
        Plot confusion matrix
        
        Args:
            dataloader: Data loader
            model_name: Name of the model
        """
        predictions, labels, _ = self.predict(dataloader)
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Similar', 'Similar'],
                   yticklabels=['Not Similar', 'Similar'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    def get_qualitative_results(self, dataloader: DataLoader, pairs: List[Tuple[str, str]], 
                              labels: List[int], num_examples: int = 10) -> Dict:
        """
        Get examples of correct and incorrect predictions
        
        Args:
            dataloader: Data loader
            pairs: List of clause pairs
            labels: True labels
            num_examples: Number of examples to return
            
        Returns:
            Dictionary with correct and incorrect examples
        """
        predictions, true_labels, probs = self.predict(dataloader)
        
        # Find correct predictions
        correct_indices = np.where(predictions == true_labels)[0]
        incorrect_indices = np.where(predictions != true_labels)[0]
        
        results = {
            'correct': [],
            'incorrect': []
        }
        
        # Sample correct examples
        if len(correct_indices) > 0:
            sample_correct = np.random.choice(correct_indices, 
                                             min(num_examples, len(correct_indices)), 
                                             replace=False)
            for idx in sample_correct:
                results['correct'].append({
                    'text1': pairs[idx][0][:200] + '...' if len(pairs[idx][0]) > 200 else pairs[idx][0],
                    'text2': pairs[idx][1][:200] + '...' if len(pairs[idx][1]) > 200 else pairs[idx][1],
                    'true_label': int(true_labels[idx]),
                    'predicted_label': int(predictions[idx]),
                    'confidence': float(probs[idx])
                })
        
        # Sample incorrect examples
        if len(incorrect_indices) > 0:
            sample_incorrect = np.random.choice(incorrect_indices,
                                              min(num_examples, len(incorrect_indices)),
                                              replace=False)
            for idx in sample_incorrect:
                results['incorrect'].append({
                    'text1': pairs[idx][0][:200] + '...' if len(pairs[idx][0]) > 200 else pairs[idx][0],
                    'text2': pairs[idx][1][:200] + '...' if len(pairs[idx][1]) > 200 else pairs[idx][1],
                    'true_label': int(true_labels[idx]),
                    'predicted_label': int(predictions[idx]),
                    'confidence': float(probs[idx])
                })
        
        return results

