"""
Mostly like the original transformer model with encoder
and masked decoder.
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        """
        :param max_len: Input length sequence.
        :param d_model: Embedding dimension.
        :param dropout: Dropout value (default=0.1)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Inputs of forward function
        :param x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        :param vocab_size: Size of vocabulary, an integer indicating
            the maximum unique words in the dataset.
        :param embed_dim: The embedding layer dimension.
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x):
        """
        :param x: Input vector.
        
        Returns:
            out: Embedding vector.
        """
        out = self.embed(x)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        :param embed_dim: Embedding dimension.
        :param n_heads = Number of attention heads. 
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert embed_dim % n_heads == 0, \
            f"Embedding dimension should be divisible by number of heads"
        self.head_dim = int(self.embed_dim / self.n_heads)

        # Query matrix (64, 64).
        self.q = nn.Linear(self.head_dim, self.head_dim)
        # Key matrix (64, 64).
        self.k = nn.Linear(self.head_dim, self.head_dim)
        # Value matrix (64, 64).
        self.v = nn.Linear(self.head_dim, self.head_dim)

        self.out = nn.Linear(self.n_heads*self.head_dim, self.embed_dim)

    def forward(self, key, query, value, mask=None):
        """
        :param key: key vector.
        :param query: query vector.
        :param value: value vector.
        :param mask: Whether masking or not, for decoder.
        """
        batch_size = key.size(0) # Batch size.
        seq_len = key.size(1) # Max. sequence length.
        inp_emb = key.size(2) # Embedding dim.
        assert inp_emb == self.embed_dim, \
            f"Input embedding {inp_emb} should match layer embedding {self.embed_dim}"
        
        seq_len_query = query.size(1)
        seq_len_value = value.size(1)

        key = key.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ) # [bs, seq_len, n_heads, head_dim] ~ [32, 1024, 8, 64]
        query = query.view(
            batch_size, seq_len_query, self.n_heads, self.head_dim
        ) # [bs, seq_len, n_heads, head_dim] ~ [32, 1024, 8, 64]
        value = value.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ) # [bs, seq_len, n_heads, head_dim] ~ [32, 1024, 8, 64]

        k = self.k(key)
        q = self.q(query)
        v = self.v(value)

        k = k.transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim]
        v = v.transpose(1, 2) # [batch_size, n_heads, seq_len, head_dim] 

        # Scaled-dot product attention.
        # Transposed key for matrix multiplication.
        k_transposed = k.transpose(-1, -2)
        dot = torch.matmul(q, k_transposed)
        if mask is not None:
            dot = dot.masked_fill(mask == 0, float('-1e20'))
        # Scaling.
        dot = dot / math.sqrt(self.head_dim) # / 64.
        scores = F.softmax(dot, dim=-1)
        # Dot product with value matix.
        scores = torch.matmul(scores, v)

        concat = scores.transpose(1,2).contiguous().view(
            batch_size, seq_len_query, self.head_dim*self.n_heads
        )

        out = self.out(concat)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()
        """
        :param embed_dim: Embedding dimension.
        :param expansion_factor: Factor determining the output dimension
            of the linear layer.
        :param n_heads: Number of attention heads.
        """ 
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor*embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor*embed_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):
        """
        :param key: Key vector.
        :param query: Query vector.
        :param value: Value vector.

        Returns:
            out: Output of the transformer block.
        """
        x = self.attention(key, query, value)
        x = x + value
        x = self.dropout1(self.norm1(x))
        ff = self.ffn(x)
        x = ff + x
        out = self.dropout2(self.norm2(x))
        return out

class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            seq_len, 
            vocab_size, 
            embed_dim, 
            num_layers=6,
            expansion_factor=4,
            n_heads=8
    ):
        """
        :param seq_len: Input sequence length.
        :param vocab_size: Number of unique tokens.
        :param embed_dim: Embedding dimension.
        :param num_layers: Number of encoder layers.
        :param expansion_factor: Factor determining the output feature
            dimension of the linear layers.
        :param n_heads: Number of attention heads.

        Returns:
            out: Transformer encoder output.
        """
        super(TransformerEncoder, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, expansion_factor, n_heads) \
            for _ in range(num_layers)]
        )

    def forward(self, x):
        x = self.embedding(x)
        out = self.positional_encoding(x)
        for layer in self.layers:
            out = layer(out, out, out) # Query, Key, Value are the same.
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        """
        :param embed_dim: Embedding dimension.
        :param exansion_factor: Factor determining the feature dimension
            of linear layers.
        :param n_heads: Number of attention heads.
        """
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(
            embed_dim, expansion_factor, n_heads
        )

    def forward(self, key, query, x, mask):
        """
        :param key: Key vector.
        :param query: Query vector.
        :param mask: Mask for multi-head attention.

        Returns:
            out: Output of the transformer block.
        """
        attended = self.attention(x, x, x, mask=mask)
        value = self.dropout(self.norm(attended + x))
        out = self.transformer_block(key, query, value)
        return out
    
class TransformerDecoder(nn.Module):
    def __init__(
            self,
            tgt_vocab_size,
            embed_dim,
            seq_len, 
            num_layers=6,
            expansion_factor=4,
            n_heads=8 
    ):
        """
        :param tgt_vocab_size: Target vocabuluary size.
        :param embed_dim: Embedding dimension.
        :param seq_len: Input sequence length.
        :param num_layers: Number of transformer layers.
        :param expansion_factor: Factor to determine the intermediate
            output feature dimension of linear layers.
        :param n_heads: Number of self attention heads.
        """
        super(TransformerDecoder, self).__init__()
        self.embedding = Embedding(tgt_vocab_size, embed_dim)
        self.postional_encoding = PositionalEncoding(seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor, n_heads) \
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embed_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        """
        :param x: Input target vector.
        :param enc_out: Encoder layer output.
        :param mask: Decoder self attention mask.

        Returns:
            out: Output vector.
        """
        x = self.embedding(x)
        x = self.postional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask)
        out = F.softmax(self.fc(x))
        return out
    
class Transformer(nn.Module):
    def __init__(
            self,
            embed_dim, 
            src_vocab_size, 
            tgt_vocab_size,
            seq_len,
            num_layers=6,
            expansion_factor=4,
            n_heads=8,
            device='cpu'
    ):
        """
        :param embed_dim: Embedding dimension.
        :param src_vocab_size: Source vocabulary size.
        :param tgt_vocab_size: Target vocabuluary size.
        :param seq_len: Input sequence lenght.
        :param num_layers: Number of transformer layers.
        :param expansion_factor: Factor to determine the intermediate
            output feature dimension of linear layers.
        :param n_heads: Number of self attention heads.
        """
        super(Transformer, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder = TransformerEncoder(
            seq_len,
            src_vocab_size,
            embed_dim,
            num_layers,
            expansion_factor,
            n_heads
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            embed_dim,
            seq_len,
            num_layers,
            expansion_factor,
            n_heads
        )
        self.device=device

    def make_tgt_mask(self, tgt):
        """
        :param tgt: Target sequence.
        
        Returns:
            tgt_mask: Target mask.
        """
        batch_size, tgt_len = tgt.shape
        # Lower triangular part of matrix filled with ones.
        tgt_mask = torch.tril(
            torch.ones((tgt_len, tgt_len))
        ).expand(batch_size, 1, tgt_len, tgt_len)
        return tgt_mask
    
    def decode(self, src, tgt):
        """
        :param src: Encoder input
        :param tgt: Decoder input

        Returns:
            out_labels: Final prediction sequence
        """
        tgt_mask = self.make_tgt_mask(tgt).to(self.device)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size, seq_len = src.shape[0], src.shape[1]
        out = tgt
        for i in range(seq_len):
            out = self.decoder(out, enc_out, tgt_mask)
            out = out[:, -1, :]
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, 0)
        return out_labels
    
    def forward(self, src, tgt):
        """
        :param src: Encoder input.
        :param tgt: Decoder input

        Returns:
            out: Output vector containing probability of each token.
        """
        tgt_mask = self.make_tgt_mask(tgt).to(self.device)
        enc_out = self.encoder(src)
        out = self.decoder(tgt, enc_out, tgt_mask)
        return out