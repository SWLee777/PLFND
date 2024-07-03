import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with optional output layer.
    """
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, embed_dim in enumerate(embed_dims):
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim if i == 0 else embed_dims[i-1], embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        if output_layer:
            self.layers.append(nn.Linear(embed_dims[-1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class CNNExtractor(nn.Module):
    """
    Convolutional Neural Network (CNN) feature extractor.
    """
    def __init__(self, feature_kernel, input_size):
        super(CNNExtractor, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(input_size, feature_num, kernel),
                nn.BatchNorm1d(feature_num),
                nn.ReLU(),
                nn.Conv1d(feature_num, feature_num, 1),  # Depth increase
                nn.BatchNorm1d(feature_num),
                nn.ReLU()
            ) for kernel, feature_num in feature_kernel.items()]
        )
        self.input_shape = sum([feature_kernel[k] for k in feature_kernel])
        self.residual_conv = nn.Conv1d(input_size, self.input_shape, 1)  # Residual connection to adjust dimensions

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        residual = self.residual_conv(share_input_data)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        residual = torch.max_pool1d(residual, residual.shape[-1])
        feature = feature + residual
        feature = feature.view(input_data.size(0), -1)
        return feature

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    """
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.dense = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.repeat(1, self.num_heads, q.size(2), 1)

        scaled_attention, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        original_size_attention = scaled_attention.view(batch_size, -1, self.embed_dim)

        output = self.dense(original_size_attention)
        return output, attn_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = k.size()[-1]
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

class MaskAttention(nn.Module):
    """
    Compute attention layer with masking support.
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores
