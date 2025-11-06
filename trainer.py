"""
Training Module
Handles model training, validation, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
import time
from tqdm import tqdm


class ClausePairDataset(Dataset):
    """Dataset class for clause pairs"""
    
    def __init__(self, pairs: List[Tuple[str, str]], labels: List[int], preprocessor):
        """
        Initialize dataset
        
        Args:
            pairs: List of (text1, text2) tuples
            labels: List of labels (0 or 1)
            preprocessor: TextPreprocessor instance
        """
        self.pairs = pairs
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        text1, text2 = self.pairs[idx]
        label = self.labels[idx]
        
        # Encode texts
        encoded1 = self.preprocessor.encode_text(text1)
        encoded2 = self.preprocessor.encode_text(text2)
        
        return {
            'text1': torch.LongTensor(encoded1),
            'text2': torch.LongTensor(encoded2),
            'label': torch.LongTensor([label])
        }


class ModelTrainer:
    """Handles training and evaluation of models"""
    
    def __init__(self, model: nn.Module, device: str = None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            text1 = batch['text1'].to(self.device)
            text2 = batch['text2'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(text1, text2)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate model
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                text1 = batch['text1'].to(self.device)
                text2 = batch['text2'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                # Forward pass
                outputs = self.model(text1, text2)
                loss = criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             num_epochs: int = 10, learning_rate: float = 0.001,
             weight_decay: float = 1e-5, patience: int = 5) -> Dict:
        """
        Train model with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            
        Returns:
            Dictionary with training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print("✓ New best model!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'training_time': training_time
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")

