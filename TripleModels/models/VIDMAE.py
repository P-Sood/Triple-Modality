import torch
import torch.nn as nn
import torch.nn.functional as F

class LMAM(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(LMAM, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True):
        batch_size = query.size(0)

        q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights