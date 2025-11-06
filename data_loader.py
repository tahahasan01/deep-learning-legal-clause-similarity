"""
Data Loading and Preprocessing Module
Handles loading CSV files and preparing data for legal clause similarity task
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import random


class LegalClauseDataLoader:
    """Loads and processes legal clause dataset"""
    
    def __init__(self, data_dir: str = "archive (1)"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
        self.clauses_by_category = defaultdict(list)
        self.all_clauses = []
        self.category_to_clauses = {}
        
    def load_all_data(self) -> Dict[str, List[str]]:
        """
        Load all CSV files and organize clauses by category
        
        Returns:
            Dictionary mapping category names to lists of clause texts
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        print(f"Loading {len(csv_files)} CSV files...")
        
        for csv_file in csv_files:
            file_path = os.path.join(self.data_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                
                # Extract category name from filename (remove .csv extension)
                category = csv_file.replace('.csv', '')
                
                # Get clause texts
                if 'clause_text' in df.columns:
                    clauses = df['clause_text'].dropna().tolist()
                    self.clauses_by_category[category] = clauses
                    self.all_clauses.extend([(clause, category) for clause in clauses])
                    
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                continue
        
        # Create reverse mapping
        for category, clauses in self.clauses_by_category.items():
            self.category_to_clauses[category] = clauses
        
        print(f"Loaded {len(self.clauses_by_category)} categories")
        print(f"Total clauses: {len(self.all_clauses)}")
        
        # Print category statistics
        category_sizes = {cat: len(clauses) for cat, clauses in self.clauses_by_category.items()}
        print(f"Categories with most clauses: {sorted(category_sizes.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        return self.clauses_by_category
    
    def create_similarity_pairs(self, num_pairs: int = None, 
                                positive_ratio: float = 0.5,
                                seed: int = 42) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Create pairs of clauses for similarity classification
        
        Args:
            num_pairs: Total number of pairs to create (None = use all possible)
            positive_ratio: Ratio of positive (similar) pairs
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (pairs, labels) where labels are 1 for similar, 0 for dissimilar
        """
        random.seed(seed)
        np.random.seed(seed)
        
        pairs = []
        labels = []
        
        # Create positive pairs (same category)
        positive_pairs_needed = int(num_pairs * positive_ratio) if num_pairs else None
        negative_pairs_needed = int(num_pairs * (1 - positive_ratio)) if num_pairs else None
        
        # Generate positive pairs
        positive_count = 0
        for category, clauses in self.clauses_by_category.items():
            if len(clauses) < 2:
                continue
                
            # Create pairs within same category
            for i in range(len(clauses)):
                for j in range(i + 1, len(clauses)):
                    if num_pairs and positive_count >= positive_pairs_needed:
                        break
                    pairs.append((clauses[i], clauses[j]))
                    labels.append(1)
                    positive_count += 1
                if num_pairs and positive_count >= positive_pairs_needed:
                    break
            if num_pairs and positive_count >= positive_pairs_needed:
                break
        
        # Generate negative pairs (different categories)
        negative_count = 0
        categories_list = list(self.clauses_by_category.keys())
        
        while (num_pairs is None) or (negative_count < negative_pairs_needed):
            # Randomly select two different categories
            cat1, cat2 = random.sample(categories_list, 2)
            
            # Randomly select clauses from each category
            if len(self.clauses_by_category[cat1]) > 0 and len(self.clauses_by_category[cat2]) > 0:
                clause1 = random.choice(self.clauses_by_category[cat1])
                clause2 = random.choice(self.clauses_by_category[cat2])
                
                pairs.append((clause1, clause2))
                labels.append(0)
                negative_count += 1
                
                if num_pairs and negative_count >= negative_pairs_needed:
                    break
        
        # Shuffle pairs
        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs, labels = zip(*combined)
        
        print(f"Created {len(pairs)} pairs: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
        
        return list(pairs), list(labels)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'num_categories': len(self.clauses_by_category),
            'total_clauses': len(self.all_clauses),
            'avg_clauses_per_category': np.mean([len(c) for c in self.clauses_by_category.values()]),
            'min_clauses_per_category': min([len(c) for c in self.clauses_by_category.values()]),
            'max_clauses_per_category': max([len(c) for c in self.clauses_by_category.values()]),
        }
        return stats

