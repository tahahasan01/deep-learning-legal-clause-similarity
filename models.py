"""
Neural Network Models for Legal Clause Similarity
Implements BiLSTM and Attention-based Encoder architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BiLSTMSimilarityModel(nn.Module):
    """
    BiLSTM-based model for clause similarity
    Uses bidirectional LSTM to encode both clauses, then compares them
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2, 
                 dropout: float = 0.3, num_classes: int = 2):
        """
        Initialize BiLSTM model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Word embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            num_classes: Number of output classes (2 for binary classification)
        """
        super(BiLSTMSimilarityModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers for similarity computation
        # BiLSTM output is 2 * hidden_dim (bidirectional)
        lstm_output_dim = 2 * hidden_dim
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Similarity computation layers
        # Input size is 2*hidden_dim because we concatenate 4 vectors of size hidden_dim//2
        combined_dim = 2 * hidden_dim  # 4 * (hidden_dim // 2) = 2 * hidden_dim
        self.similarity_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, text1: torch.Tensor, text2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            text1: First clause batch [batch_size, seq_length]
            text2: Second clause batch [batch_size, seq_length]
            
        Returns:
            Logits for similarity classification [batch_size, num_classes]
        """
        # Embed both texts
        emb1 = self.embedding(text1)  # [batch_size, seq_length, embedding_dim]
        emb2 = self.embedding(text2)
        
        # Pass through BiLSTM
        lstm_out1, _ = self.lstm(emb1)  # [batch_size, seq_length, 2*hidden_dim]
        lstm_out2, _ = self.lstm(emb2)
        
        # Use last hidden state (or mean pooling)
        # Option 1: Use last timestep
        # rep1 = lstm_out1[:, -1, :]  # [batch_size, 2*hidden_dim]
        # rep2 = lstm_out2[:, -1, :]
        
        # Option 2: Mean pooling (often works better)
        rep1 = torch.mean(lstm_out1, dim=1)  # [batch_size, 2*hidden_dim]
        rep2 = torch.mean(lstm_out2, dim=1)
        
        # Project to same dimension
        proj1 = self.projection(rep1)  # [batch_size, hidden_dim//2]
        proj2 = self.projection(rep2)
        
        # Compute similarity features
        # Concatenate, subtract, and multiply (common similarity features)
        diff = torch.abs(proj1 - proj2)  # [batch_size, hidden_dim//2]
        mult = proj1 * proj2  # [batch_size, hidden_dim//2]
        combined = torch.cat([proj1, proj2, diff, mult], dim=1)  # [batch_size, 2*hidden_dim]
        
        # Final classification
        output = self.similarity_layers(combined)  # [batch_size, num_classes]
        
        return output


class AttentionEncoderSimilarityModel(nn.Module):
    """
    Attention-based Encoder model for clause similarity
    Uses self-attention and cross-attention mechanisms
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 2):
        """
        Initialize Attention-based model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Word embedding dimension
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super(AttentionEncoderSimilarityModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Positional encoding (simple learned embeddings)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, embedding_dim))
        
        # Multi-head self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cross-attention layer (to attend between the two clauses)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),  # 4 = 2 clauses * 2 (mean + max pooling)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, text1: torch.Tensor, text2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanisms
        
        Args:
            text1: First clause batch [batch_size, seq_length]
            text2: Second clause batch [batch_size, seq_length]
            
        Returns:
            Logits for similarity classification [batch_size, num_classes]
        """
        batch_size, seq_len1 = text1.shape
        _, seq_len2 = text2.shape
        
        # Embed both texts
        emb1 = self.embedding(text1)  # [batch_size, seq_length, embedding_dim]
        emb2 = self.embedding(text2)
        
        # Add positional encoding
        emb1 = emb1 + self.pos_encoding[:, :seq_len1, :]
        emb2 = emb2 + self.pos_encoding[:, :seq_len2, :]
        
        # Create attention mask for padding
        mask1 = (text1 != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        mask2 = (text2 != 0).unsqueeze(1).unsqueeze(2)
        
        # Self-attention encoding
        encoded1 = self.encoder(emb1, src_key_padding_mask=(text1 == 0))  # [batch_size, seq_len1, embedding_dim]
        encoded2 = self.encoder(emb2, src_key_padding_mask=(text2 == 0))  # [batch_size, seq_len2, embedding_dim]
        
        # Cross-attention: text1 attends to text2 and vice versa
        cross_attn1, _ = self.cross_attention(
            encoded1, encoded2, encoded2,
            key_padding_mask=(text2 == 0)
        )  # [batch_size, seq_len1, embedding_dim]
        
        cross_attn2, _ = self.cross_attention(
            encoded2, encoded1, encoded1,
            key_padding_mask=(text1 == 0)
        )  # [batch_size, seq_len2, embedding_dim]
        
        # Pooling: mean and max pooling
        # For text1
        mean1 = torch.mean(cross_attn1, dim=1)  # [batch_size, embedding_dim]
        max1, _ = torch.max(cross_attn1, dim=1)  # [batch_size, embedding_dim]
        
        # For text2
        mean2 = torch.mean(cross_attn2, dim=1)  # [batch_size, embedding_dim]
        max2, _ = torch.max(cross_attn2, dim=1)  # [batch_size, embedding_dim]
        
        # Combine features
        combined = torch.cat([mean1, max1, mean2, max2], dim=1)  # [batch_size, embedding_dim * 4]
        
        # Final classification
        output = self.classifier(combined)  # [batch_size, num_classes]
        
        return output


class ESIMSimilarityModel(nn.Module):
    """
    Enhanced Sequential Inference Model (ESIM) for clause similarity
    Specifically designed for semantic similarity tasks with local inference and composition
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 1,
                 dropout: float = 0.3, num_classes: int = 2):
        """
        Initialize ESIM model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Word embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super(ESIMSimilarityModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Input encoding: BiLSTM
        self.input_encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Inference composition: BiLSTM
        # Input: 4 features (original, aligned, diff, mult) = 4 * 2*hidden_dim
        self.composition = nn.LSTM(
            hidden_dim * 2 * 4,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier
        # Output from composition is 2 * hidden_dim (bidirectional)
        # We use max and mean pooling, so 2 * 2 * hidden_dim = 4 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, text1: torch.Tensor, text2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ESIM architecture
        
        Args:
            text1: First clause batch [batch_size, seq_length]
            text2: Second clause batch [batch_size, seq_length]
            
        Returns:
            Logits for similarity classification [batch_size, num_classes]
        """
        # Embed both texts
        emb1 = self.embedding(text1)  # [batch_size, seq_len1, embedding_dim]
        emb2 = self.embedding(text2)  # [batch_size, seq_len2, embedding_dim]
        
        # Input encoding: BiLSTM
        encoded1, _ = self.input_encoder(emb1)  # [batch_size, seq_len1, 2*hidden_dim]
        encoded2, _ = self.input_encoder(emb2)  # [batch_size, seq_len2, 2*hidden_dim]
        
        # Local inference: Soft attention alignment
        # Attention matrix: [batch_size, seq_len1, seq_len2]
        attention_matrix = torch.bmm(encoded1, encoded2.transpose(1, 2))
        
        # Create mask for padding
        mask1 = (text1 != 0).unsqueeze(2).float()  # [batch_size, seq_len1, 1]
        mask2 = (text2 != 0).unsqueeze(1).float()  # [batch_size, 1, seq_len2]
        attention_mask = mask1 * mask2  # [batch_size, seq_len1, seq_len2]
        
        # Apply mask (set padding to very negative value)
        attention_matrix = attention_matrix.masked_fill(attention_mask == 0, -1e9)
        
        # Soft attention weights
        attn_weights_1_to_2 = F.softmax(attention_matrix, dim=2)  # [batch_size, seq_len1, seq_len2]
        attn_weights_2_to_1 = F.softmax(attention_matrix.transpose(1, 2), dim=2)  # [batch_size, seq_len2, seq_len1]
        
        # Aligned representations
        aligned1 = torch.bmm(attn_weights_1_to_2, encoded2)  # [batch_size, seq_len1, 2*hidden_dim]
        aligned2 = torch.bmm(attn_weights_2_to_1, encoded1)  # [batch_size, seq_len2, 2*hidden_dim]
        
        # Enhanced local inference: combine original and aligned
        enhanced1 = torch.cat([
            encoded1,
            aligned1,
            encoded1 - aligned1,  # difference
            encoded1 * aligned1   # element-wise product
        ], dim=2)  # [batch_size, seq_len1, 4 * 2*hidden_dim]
        
        enhanced2 = torch.cat([
            encoded2,
            aligned2,
            encoded2 - aligned2,
            encoded2 * aligned2
        ], dim=2)  # [batch_size, seq_len2, 4 * 2*hidden_dim]
        
        # Inference composition: BiLSTM
        composed1, _ = self.composition(enhanced1)  # [batch_size, seq_len1, 2*hidden_dim]
        composed2, _ = self.composition(enhanced2)  # [batch_size, seq_len2, 2*hidden_dim]
        
        # Pooling: mean and max
        mean1 = torch.mean(composed1, dim=1)  # [batch_size, 2*hidden_dim]
        max1, _ = torch.max(composed1, dim=1)  # [batch_size, 2*hidden_dim]
        mean2 = torch.mean(composed2, dim=1)  # [batch_size, 2*hidden_dim]
        max2, _ = torch.max(composed2, dim=1)  # [batch_size, 2*hidden_dim]
        
        # Combine all features
        combined = torch.cat([mean1, max1, mean2, max2], dim=1)  # [batch_size, 4 * 2*hidden_dim]
        
        # Final classification
        output = self.classifier(combined)  # [batch_size, num_classes]
        
        return output

