In this repo, we'll follow the karpathy tutorial to do the gpt2 124M parameters

the paper name is "Language Models are Unsupervised Multitask Learners" and the link is https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

each model that gets released is part of a miniseries of various sizes.
The biggest model is called the gpt

now we're going to do a smaller model.

the model will be less good because power and accuracy increases as size increases.

hde 124M model has 12 layers and 768 channels (or dimensions)

to test how the model is doing, we're going to have a loss function. If we see that we go from doing that task not very well to very well, then it works.

we also have all the weights for gpt2 because open ai released them.

We will use the gpt-3 paper as well because it mentions a lot of very valuable information. The link is https://arxiv.org/pdf/2005.14165.pdf

if we look at the openai repo, we see that it's using tenserflow, which is not as good as pytorch.

we'll use pytorch instead and we'll use the huggingface transformers library.

`https://github.com/openai/gpt-2` this is the link to the openai repo

`https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt2`

and this is the link to the transformers

`https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py`

this is the link mentionned at 4:26


so i was struglling with the notebook thing but turns out that the text that i saw below the code is actually the output of the code.

wte means word token embeddings

transformer.wte.weight torch.Size([50257, 768]) 
gpt 2 tokenizer has 50257 tokens and each token has 768 dimensions
we have a 768 dimensional embedding that is the distributed representation that stands in for that token.

each token is a stringc piece and each of the 768 numbers is a vector that represents that token.

transformer.wpe.weight torch.Size([1024, 768]) 
this is the lookup table for the position embeddings
gpt 2 has a maximum sequence length of 1024 tokens 
we have up to 1024 positions that each token can be attending to in the past
every one of these positions has has a fixed vector of 768 that is learned by optimization

everything below these two lines 
transformer.wte.weight torch.Size([50257, 768]) 
transformer.wpe.weight torch.Size([1024, 768])

is just other weights and and biases of the transformer.
ok so there was an error with numpy, what i actually needed to do was to use a previous version

`pip install numpy==1.24.1`

the  graph obtained in 
plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300,:300], cmap="gray")

is a graph of the attention weights of the first head of the first layer of the transformer
if one is into mechanistic interpretability, one could try and interpret that graph, but for us what is important right now is that this is before training and the weights are random. 

