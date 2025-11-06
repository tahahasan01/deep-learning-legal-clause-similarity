"""
Text Preprocessing Module
Handles tokenization, vocabulary building, and text encoding
"""

import re
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict
import pickle


class TextPreprocessor:
    """Preprocesses legal clause text for neural network input"""
    
    def __init__(self, max_vocab_size: int = 10000, max_seq_length: int = 200):
        """
        Initialize preprocessor
        
        Args:
            max_vocab_size: Maximum vocabulary size
            max_seq_length: Maximum sequence length for padding/truncation
        """
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()\[\]{}"\'-]', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        text = self.clean_text(text)
        # Simple word tokenization (split on whitespace)
        tokens = text.split()
        return tokens
    
    def build_vocabulary(self, texts: List[str]):
        """
        Build vocabulary from training texts
        
        Args:
            texts: List of text strings
        """
        print("Building vocabulary...")
        word_counts = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # Get most common words
        most_common = word_counts.most_common(self.max_vocab_size - 4)  # Reserve space for special tokens
        
        # Build vocabulary
        self.word_to_idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        
        idx = 4
        for word, count in most_common:
            self.word_to_idx[word] = idx
            idx += 1
        
        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common words: {list(word_counts.most_common(10))}")
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of indices
        
        Args:
            text: Input text
            
        Returns:
            List of word indices
        """
        tokens = self.tokenize(text)
        sequence = []
        
        for token in tokens:
            if token in self.word_to_idx:
                sequence.append(self.word_to_idx[token])
            else:
                sequence.append(self.word_to_idx[self.UNK_TOKEN])
        
        return sequence
    
    def pad_sequence(self, sequence: List[int]) -> List[int]:
        """
        Pad or truncate sequence to fixed length
        
        Args:
            sequence: Input sequence
            
        Returns:
            Padded/truncated sequence
        """
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            sequence = sequence + [self.word_to_idx[self.PAD_TOKEN]] * (self.max_seq_length - len(sequence))
        
        return sequence
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to numpy array
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of shape (max_seq_length,)
        """
        sequence = self.text_to_sequence(text)
        padded = self.pad_sequence(sequence)
        return np.array(padded, dtype=np.int32)
    
    def encode_pairs(self, pairs: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode pairs of texts
        
        Args:
            pairs: List of (text1, text2) tuples
            
        Returns:
            Tuple of (encoded_text1, encoded_text2) arrays
        """
        encoded1 = []
        encoded2 = []
        
        for text1, text2 in pairs:
            encoded1.append(self.encode_text(text1))
            encoded2.append(self.encode_text(text2))
        
        return np.array(encoded1), np.array(encoded2)
    
    def save(self, filepath: str):
        """Save preprocessor to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'vocab_size': self.vocab_size,
                'max_vocab_size': self.max_vocab_size,
                'max_seq_length': self.max_seq_length
            }, f)
    
    def load(self, filepath: str):
        """Load preprocessor from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.vocab_size = data['vocab_size']
            self.max_vocab_size = data['max_vocab_size']
            self.max_seq_length = data['max_seq_length']

