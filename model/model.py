import torch
import torch.nn as nn
from torch.nn import functional as F
from config.config import *
from model.utilities import gelu_func, LayerNorm
import math

if device_comp:
    device= torch.device(device_comp)

class AttentionHead(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch, time, single_embed_size = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * single_embed_size ** -0.5
        masked_output = wei.masked_fill(self.tril[:time, :time] == 0, float('-inf'))
        masked_softmax = F.softmax(masked_output, dim=-1)
        masked_softmax = self.dropout(masked_softmax)
        output = masked_softmax @ v
        return output
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.multiHeadAtt = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)]) 
        self.projections = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):

        attention_list = [attn_head(x) for attn_head in self.multiHeadAtt]
        attention_list = torch.cat(attention_list, dim = -1)
        return self.dropout(self.projections(attention_list))
    

class causalSelfAttention(nn.Module):

    """
    merging selfAttention and MultiHeadAttention
    """
    def __init__(self):
        super().__init__()
        assert n_embed % num_heads == 0
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        batch, time, single_embed_size = x.shape
        q, k, v = self.c_attn(x).split(n_embed, dim = 2)
        k = k.view(batch, time, num_heads, single_embed_size // num_heads).transpose(1, 2)
        q = q.view(batch, time, num_heads, single_embed_size // num_heads).transpose(1, 2)
        v = v.view(batch, time, num_heads, single_embed_size // num_heads).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                                 dropout_p=dropout if mode == 'training' else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, time, single_embed_size)
        y = self.resid_dropout(self.c_proj(y))
        return y
    
    

class SimpleFeedForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed)
        self.c_proj = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = gelu_func(x)
        x = self.c_proj(x)
        return self.dropout(x)
    

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.att_head = causalSelfAttention()
        self.ffwd = SimpleFeedForward()
        self.layer_norm1 = LayerNorm(n_embed, bias)
        self.layer_norm2 = LayerNorm(n_embed, bias)

    def forward(self, x):
        x = x + self.att_head(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x
        



class GPTModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.block = nn.Sequential(*[TransformerBlock() for _ in range(n_layer)])
        self.layer_norm3 = nn.LayerNorm(n_embed, bias)
        self.dropout = nn.Dropout(dropout)
        self.apply(self.__init_weights__)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weights'):
                torch.nn.init.normal_(p, mean = 0.0, std = 0.02 / math.sqrt(2 * n_layer))
        
        
    def __init_weights__(self, module):    
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            
            
    def forward(self, idx, target = None):

        B, T = idx.shape
        device_in = idx.device
        token_embeddings = self.token_embeddings(idx)
        positional_embeddings = self.position_embeddings(torch.arange(T, dtype = torch.long, device=device_in))
        x = token_embeddings + positional_embeddings
        x = self.layer_norm3(self.block(x))

        logits = self.lm_head(x)
        
        if target is None:
            loss = None
        else:
            batch, block, channel = logits.shape
            logits = logits.view(batch * block, channel)
            target = target.view(batch * block)
            loss = F.cross_entropy(logits, target)

        return logits, loss
    
    def generate_captions(self, idx, max_tokens):
        for tok in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat([idx, idx_next], dim = 1)
        return idx