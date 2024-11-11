from dataclasses import dataclass # dataclasses module provides a decorator and functions for automatically adding special methods such as __init__() and __repr__() to user-defined classes

import torch # import the torch module
import torch.nn as nn # import the torch.nn module
from torch.nn import functional as F # import the torch.nn.functional module
import math


# ------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config): # define the __init__ method that takes in the self and config parameters
        super().__init__() # call the __init__ method of the parent class
        assert config.n_embd % config.n_head == 0 # assert that the embedding size is divisible by the number of heads
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # c_attn is a linear layer that converts the input into 3 times the embedding size
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # c_proj is a linear layer that converts the output into the embedding size
        # regularization
        self.n_head = config.n_head # n_head is the number of heads
        self.n_embd = config.n_embd # n_embd is the embedding size

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # register a buffer called bias, it's a lower triangular matrix of ones

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # attention calculation, scaled by the sqrt of the dimension of the key
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # mask out attention for padding tokens
        att = F.softmax(att, dim=-1) # apply softmax to the attention scores
        y = att @ v # multiply the attention scores by the values
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y) # apply the output projection
        return y
# ------------------------------------------------------------------
class MLP(nn.Module): # define a class called MLP that inherits from nn.Module
    def __init__(self, config): # define the __init__ method that takes in the self and config parameters
        super().__init__() # call the __init__ method of the parent class
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # c_fc is a linear layer that converts the input into 4 times the embedding size
        self.gelu = nn.GELU(approximate='tanh') # gelu is the GELU activation function, it's a Gaussian Error Linear Unit function that's used as an activation function. 'tanh' is the approximation method
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # c_proj is a linear layer that converts the output of the GELU activation function into the embedding size

        # in these two lines above, we use 4 because it's a common practice in the transformer architecture to use 4 times the embedding size in the feedforward network

    def forward(self, x): # define the forward method that takes in the self and x parameters
        x = self.c_fc(x) # x is passed through the first linear layer
        x = self.gelu(x) # x is passed through the GELU activation function
        x = self.c_proj(x) # x is passed through the second linear layer
        return x # return the output, the type is torch.Tensor

# ------------------------------------------------------------------

class Block(nn.Module): # define a class called Block that inherits from nn.Module

    def __init__(self, config): # define the __init__ method that takes in the self and config parameters
        super().__init__() # call the __init__ method of the parent class
        self.ln_1 = nn.LayerNorm(config.n_embd) # ln_1 is a layer normalization layer, config.n_embd is the embedding size
        self.attn = CausalSelfAttention(config) # attn is a causal self-attention layer, it takes in the config parameter
        self.ln_2 = nn.LayerNorm(config.n_embd) # ln_2 is a layer normalization layer, it's the second one
        self.mlp = MLP(config) # mlp is a multi-layer perceptron layer, it takes in the config parameter

    def forward(self, x): # define the forward method that takes in the self and x parameters
        x = x + self.attn(self.ln_1(x)) # x is the input, it's passed through the first layer normalization layer, then through the causal self-attention layer, and added to the input
        x = x + self.mlp(self.ln_2(x)) # x is passed through the second layer normalization layer, then through the multi-layer perceptron layer, and added to the input
        return x # return the output, which looks like the input but with some transformations applied, the type is torch.Tensor


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of transformer layers
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict( # nn.ModuleDict is a dictionary that registers all the modules it contains
            wte = nn.Embedding(config.vocab_size, config.n_embd), # wte is the word token embedding, which is an embedding layer that converts the input tokens into embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # wpe is the word position embedding, which is an embedding layer that converts the input positions into embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # h is a list of transformer blocks
            ln_f = nn.LayerNorm(config.n_embd), # ln_f is a layer normalization layer
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # lm_head is a linear layer that converts the transformer output into logits, it's the final classification layer

    def forward(self, idx): # define the forward method that takes in the self and idx parameters
        # idx is of shape (B, T)
        B, T = idx.size() # batch size, sequence length
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # logits of shape (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

# ------------------------------------------------------------------
num_return_sequences = 5
max_length = 30

# load the model
model = GPT.from_pretrained('gpt2') # create a GPT model from the pretrained GPT-2 model
print("didn't crash! yo!")
model = GPT.from_pretrained('gpt2') # create a GPT model from the pretrained GPT-2 model
model.eval()
model.to('cuda')

print("did not crash after moving to cuda! yoho!")

# prefix token

import tiktoken

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('cuda')

print("didn't crash after token encoding! yohoho!")

# generate ! right now x is (B, T) where B is 5 and T is 8
# set the seed to 42

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)