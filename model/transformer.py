#!/usr/bin/env python3
"""
Simple Transformer model implementation for language modeling.
"""

import tensorflow as tf
import logging
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        pe = tf.zeros((max_len, d_model))
        position = tf.range(0, max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / d_model))
        
        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.stack([
                tf.repeat(tf.range(max_len), tf.size(tf.range(0, d_model, 2))),
                tf.tile(tf.range(0, d_model, 2), [max_len])
            ], axis=1),
            tf.reshape(tf.sin(position * div_term), [-1])
        )
        
        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.stack([
                tf.repeat(tf.range(max_len), tf.size(tf.range(1, d_model, 2))),
                tf.tile(tf.range(1, d_model, 2), [max_len])
            ], axis=1),
            tf.reshape(tf.cos(position * div_term), [-1])
        )
        
        self.pe = pe[tf.newaxis, ...]
    
    def call(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:, :tf.shape(x)[1], :]


class TransformerLM(tf.keras.Model):
    """Transformer language model."""
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, dropout_rate=0.1, max_len=5000):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Transformer layers
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            )
        
        # Output layer
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, training=False, mask=None):
        """Forward pass."""
        # Get sequence length
        seq_len = tf.shape(x)[1]
        
        # Embedding and positional encoding
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Create lookahead mask for autoregressive property in training
        if mask is None:
            mask = create_look_ahead_mask(seq_len)
        
        # Pass through transformer blocks - use keyword arguments
        for block in self.transformer_blocks:
            x = block(x, training=training, mask=mask)
        
        # Final linear layer
        logits = self.final_layer(x)
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
        })
        return config


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with self-attention."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()
        
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model//num_heads
        )
        self.ffn = self.point_wise_feed_forward_network(d_model, d_ff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def point_wise_feed_forward_network(self, d_model, d_ff):
        """Feed forward network."""
        return tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
    
    def call(self, x, training=False, mask=None):
        """Forward pass for transformer block."""
        # Multi-head attention - use keyword arguments
        attn_output = self.attention(
            query=x, value=x, key=x, 
            attention_mask=mask, training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


def create_look_ahead_mask(size):
    """Create a look-ahead mask to prevent attending to future tokens."""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (size, size)


def create_transformer_lm(vocab_size, d_model=512, num_heads=8, num_layers=6,
                         d_ff=2048, dropout_rate=0.1, max_len=5000):
    """Create a transformer language model."""
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        max_len=max_len
    )
    
    logger.info(f"Created transformer model with {num_layers} layers, {d_model} dimensions, {num_heads} heads")
    return model


class PerplexityMetric(tf.keras.metrics.Mean):
    """Perplexity metric for language modeling."""
    
    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update perplexity state with cross entropy loss."""
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        perplexity = tf.exp(tf.reduce_mean(cross_entropy))
    
        super().update_state([perplexity], sample_weight)