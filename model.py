"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # parameters related to number of heads, block size, etc
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_embd_head = config.n_embd // config.n_head
        self.block_size = config.block_size
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        # rotary positional encoding if using
        self.use_rope = config.use_rope
        if self.use_rope:
            self.create_rope_cache()
    
    def create_rope_cache(self,base=10_000):
        dim=self.n_embd_head
        theta = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(base)/ dim))
        self.register_buffer("theta", theta, persistent=False)
        seq_idx = torch.arange(self.block_size, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        rope_cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("rope_cache", rope_cache, persistent=False)

    def apply_rope(self,x,input_pos=None):
        seq_len = x.size(1)
        # extract the values based on whether input_pos is set or not
        rope_cache_ = self.rope_cache[:seq_len] if input_pos is None else self.cache[input_pos]
        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache_ = rope_cache_.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache_[..., 0]
                - xshaped[..., 1] * rope_cache_[..., 1],
                xshaped[..., 1] * rope_cache_[..., 0]
                + xshaped[..., 0] * rope_cache_[..., 1],
            ],
            -1,
        )
        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        if self.use_rope:
            kT = k.view(B, T, self.n_head, self.n_embd_head)
            qT = q.view(B, T, self.n_head, self.n_embd_head)
            kT = self.apply_rope(kT)
            qT = self.apply_rope(qT)
            k = kT.transpose(1, 2)
            q = qT.transpose(1, 2)
            v = v.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2)
        else:
            k = k.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.n_embd_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            #if mask is not None:
            #    expanded_mask = mask[:,None,None,:]
            #    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=expanded_mask, dropout_p=self.dropout if self.training else 0, is_causal=True)
            #else:
            #    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            if mask is not None:
                attn_mask = torch.zeros(B,T,T,dtype=q.dtype).to(q)
                causal_mask = torch.ones(T,T).tril(diagonal=0).unsqueeze(0).to(q)
                full_mask = (mask.transpose(1,2) * causal_mask).bool() # (B,1,T) * (1,T,T) -> (B,T,T)
                attn_mask = attn_mask.masked_fill(full_mask.logical_not(),float("-inf"))[:,None,:,:] # add additional axis to broadcast to nhead
                #print(mask.shape)
                #print(attn_mask.shape)
                #print(q.shape,k.shape,v.shape)
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
                
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x),mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # maximum sequence length? 
    input_dim: int = 1 # default to 1d time-series data
    context_dim: int = 3 # dimension of context vector (initial conditions)
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 64 # embedding dimensions
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_pe: bool = False # True = use positional encoding
    use_rope: bool = True # True = use rotary positional embedding (RoPE)
    tokenized: bool = False # False - only use if we want to tokenize real numbers
    vocab_size: int = 1024

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.input_dim is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd) if config.tokenized else nn.Linear(config.input_dim, config.n_embd, bias=False),
            #wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            #ctxe = nn.Linear(config.context_dim,config.n_embd)
        ))
        if config.tokenized:
            self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
            self.transformer.wte.weight = self.lm_head.weight # weight tying
        else:
            self.lm_head = nn.Linear(config.n_embd, config.input_dim, bias=False)

        # configure positional embedding if desired
        if config.use_rope and config.use_pe:
            print("Error - shouldn't use RoPE and vanilla PE at the same time!")
            sys.exit()
        if config.use_pe:
            print("Using vanilla positional embedding")
            self.transformer["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        if config.use_rope:
            print("Using RoPE")

        # configure to take context if desired
        if config.context_dim is not None and config.context_dim> 0:
            self.use_context = True
            self.transformer['ctxe'] = nn.Linear(config.context_dim,config.n_embd)
        else:
            self.use_context = False
        
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        #self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.use_pe:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, context=None, mask=None):
        # idx has shape (b, t, n_embd), context has shape (b, n_ctx)
        device = idx.device
        if self.config.tokenized:
            b, t = idx.size()
        else:
            b, t, n_inp = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t+1, dtype=torch.long, device=device) # shape (t+1) to account for additional context embedding

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.use_context:
            ctx_emb = self.transformer.ctxe(context).unsqueeze(1) # context embeddings of shape (b, 1, n_embd)
            tok_emb = torch.cat([ctx_emb,tok_emb],dim=1) # full input of shape (b, t+1, n_embd)
        
        # add vanilla positional embedding if using (if using RoPE, done at the level of the self-attention blocks)
        if self.config.use_pe:
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        
        # prepare mask if using
        if mask is not None and self.use_context:
            # pad the mask to account for additional context token at the front
            extra = torch.ones((idx.shape[0],1),dtype=torch.bool).to(device)
            padded_mask = torch.cat((extra,mask),dim=1)
        else:
            padded_mask = mask
        
        for block in self.transformer.h:
            x = block(x,mask=padded_mask)
        x = self.transformer.ln_f(x)

        # project outputs
        x = self.lm_head(x)
        return x[:,1:,:] if self.use_context else x # don't return the extra token we added for the context vector

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.config.use_pe:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
