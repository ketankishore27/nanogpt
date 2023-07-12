from model.model import SimpleFeedForward, TransformerBlock
from config.config import *
import pytorch_lightning as pl
import numpy as np
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import tiktoken
from model.utilities import LayerNorm
import inspect


class lightning_GPTModel_wrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.block = nn.Sequential(*[TransformerBlock() for _ in range(n_layer)])
        self.layer_norm3 = LayerNorm(n_embed, bias)
        self.dropout = nn.Dropout(dropout)
        self.encoder = tiktoken.get_encoding('gpt2')
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
            
    
    def forward(self, idx, training=None):
        block, time_step = idx.shape
        device_in = idx.device
        token_embeddings = self.token_embeddings(idx)
        positional_embeddings = self.position_embeddings(torch.arange(time_step, dtype = torch.long, device=device_in))
        x = token_embeddings + positional_embeddings
        x = self.dropout(x)
        x = self.layer_norm3(self.block(x))
        if training:
            return self.lm_head(x)
        else:
            return self.lm_head(x[:, [-1], :])
    
    
    def training_step(self, batch, batch_idx):
        input_seq, target = batch
        logits = self(input_seq, training=True)
        batch, block, channel = logits.shape
        logits = logits.view(-1, logits.size(-1))#(batch * block, channel)
        target_new = target.view(-1)#(batch * block)
        loss = F.cross_entropy(logits, target_new, ignore_index=-1)
        self.log("training_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        input_seq, target = batch
        device_in = input_seq.device
        logits = self(input_seq, training=True)
        batch, block, channel = logits.shape
        logits = logits.view(-1, logits.size(-1))#(batch * block, channel)
        target_new = target.view(-1)#(batch * block)
        loss = F.cross_entropy(logits, target_new, ignore_index=-1)
        print("\nValidation Loss: {}\n".format(loss)) 
        print(self.encoder.decode(self.generate_captions(torch.tensor([self.encoder.encode_ordinary(" ")], 
                                                                      dtype = torch.long).to(device_in), 20)[0].tolist()))
        print("\n ---------  \n")
        self.log("validation_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)


    def configure_optimizers_custom(self):
        return torch.optim.AdamW(self.parameters(), lr = learning_rate)
    

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_param = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_param = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_group = [
            {'params': decay_param, 'weight_decay': weight_decay},
            {'params': no_decay_param, 'weight_decay': 0.0}
        ]
        num_decay_param = sum([p.numel() for p in decay_param])
        num_no_decay_param = sum([p.numel() for p in no_decay_param])
        print("Number of Decay parameters: {}".format(num_decay_param))
        print("Number of Non-Decay parameters: {}".format(num_no_decay_param))
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        #use_fused = fused_available and device_comp in ['cuda', 'mps']
        #extra_args = dict(fused=True) if use_fused else dict()
        extra_args = dict()
        optimizer = torch.optim.AdamW(optim_group, lr=min_lr, betas=((beta1, beta2)), **extra_args)
        #print(f"using fused AdamW: {use_fused}")
        return optimizer
            
    
    @torch.no_grad()
    def generate_captions(self, idx, max_tokens):
        for tok in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = 1)
            print(probs)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat([idx, idx_next], dim = 1)
        return idx
