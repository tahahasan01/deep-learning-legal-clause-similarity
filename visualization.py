"""
Visualization Utilities
Creates training graphs and result visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns


class TrainingVisualizer:
    """Creates visualizations for training progress and results"""
    
    @staticmethod
    def plot_training_history(history: Dict, model_name: str = "Model", save_path: str = None):
        """
        Plot training and validation loss/accuracy curves
        
        Args:
            history: Dictionary with 'train_losses', 'val_losses', 'train_accuracies', 'val_accuracies'
            model_name: Name of the model
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Plot loss
        axes[0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(epochs, history['train_accuracies'], 'b-', label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_accuracies'], 'r-', label='Val Accuracy', linewidth=2)
        axes[1].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_model_comparison(metrics_list: List[Dict], model_names: List[str], save_path: str = None):
        """
        Compare multiple models using bar charts
        
        Args:
            metrics_list: List of metric dictionaries
            model_names: List of model names
            save_path: Path to save the figure
        """
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 5))
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [m[metric] for m in metrics_list]
            
            bars = axes[idx].bar(model_names, values, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{metric.upper().replace("_", "-")}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=11)
            axes[idx].set_ylim([0, 1.1])
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}',
                             ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison figure saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_qualitative_results(qualitative_results: Dict, model_name: str = "Model", 
                                save_path: str = None):
        """
        Display qualitative results (correct and incorrect examples)
        
        Args:
            qualitative_results: Dictionary with 'correct' and 'incorrect' lists
            model_name: Name of the model
            save_path: Path to save the figure
        """
        num_correct = len(qualitative_results['correct'])
        num_incorrect = len(qualitative_results['incorrect'])
        
        if num_correct == 0 and num_incorrect == 0:
            print("No qualitative results to display")
            return None
        
        fig, axes = plt.subplots(max(num_correct, num_incorrect), 2, 
                                 figsize=(20, max(num_correct, num_incorrect) * 3))
        
        if num_correct == 0 or num_incorrect == 0:
            axes = axes.reshape(-1, 2) if isinstance(axes, np.ndarray) else [[axes]]
        
        # Plot correct examples
        for idx, example in enumerate(qualitative_results['correct'][:num_correct]):
            if num_correct > 1:
                ax = axes[idx, 0]
            else:
                ax = axes[0] if isinstance(axes, np.ndarray) else axes
            
            ax.text(0.1, 0.9, f"Correct Prediction (Confidence: {example['confidence']:.3f})", 
                   transform=ax.transAxes, fontsize=12, fontweight='bold', 
                   verticalalignment='top')
            ax.text(0.1, 0.7, f"Text 1: {example['text1']}", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   wrap=True)
            ax.text(0.1, 0.4, f"Text 2: {example['text2']}", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   wrap=True)
            ax.axis('off')
        
        # Plot incorrect examples
        for idx, example in enumerate(qualitative_results['incorrect'][:num_incorrect]):
            if num_incorrect > 1:
                ax = axes[idx, 1]
            else:
                ax = axes[1] if isinstance(axes, np.ndarray) else axes
            
            ax.text(0.1, 0.9, f"Incorrect Prediction (Confidence: {example['confidence']:.3f})", 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', color='red')
            ax.text(0.1, 0.7, f"Text 1: {example['text1']}", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   wrap=True)
            ax.text(0.1, 0.4, f"Text 2: {example['text2']}", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   wrap=True)
            ax.axis('off')
        
        plt.suptitle(f'{model_name} - Qualitative Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Qualitative results saved to {save_path}")
        
        return fig

