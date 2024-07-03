import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

class FeedForwardNN(nn.Module):
    """
    A simple feed-forward neural network with optional dropout and layer normalization.
    """
    def __init__(self, in_dim, hidden_layers, dropout_rate, include_output=True):
        super().__init__()
        self.network = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_layers):
            self.network.append(nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_layers[i-1], hidden_dim),
                nn.LayerNorm(hidden_dim),  # Using LayerNorm instead of BatchNorm
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
        if include_output:
            self.network.append(nn.Linear(hidden_layers[-1], 1))

    def forward(self, x):
        """
        Forward pass through the network.
        """
        for layer in self.network:
            x = layer(x)
        return x

class ConvFeatureExtractor(nn.Module):
    """
    A convolutional feature extractor with multiple convolutional layers and max pooling.
    """
    def __init__(self, kernel_to_features, in_channels):
        super(ConvFeatureExtractor, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for kernel_size, num_features in kernel_to_features.items():
            self.conv_layers.append(nn.Conv1d(in_channels, num_features, kernel_size))
            self.batch_norms.append(nn.BatchNorm1d(num_features))

    def forward(self, x):
        """
        Forward pass through the convolutional layers with activation, batch normalization, and max pooling.
        """
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_channels, sequence_length)
        feature_list = []

        for conv, bn in zip(self.conv_layers, self.batch_norms):
            conv_out = conv(x)
            bn_out = bn(F.relu(conv_out))
            pooled_out = torch.max_pool1d(bn_out, bn_out.shape[-1]).squeeze(-1)
            feature_list.append(pooled_out)

        return torch.cat(feature_list, dim=1)

class AttentionLayer(nn.Module):
    """
    Computes attention weights and applies them to the inputs.
    """
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        """
        Forward pass to compute attention scores and apply them to the inputs.
        """
        attn_scores = self.attention_fc(x).view(-1, x.size(1))  # Compute raw attention scores
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))  # Mask padding tokens
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(1)  # Normalize scores to probabilities
        weighted_sum = torch.matmul(attn_weights, x).squeeze(1)  # Compute weighted sum of inputs
        return weighted_sum, attn_weights
