`https://youtu.be/l8pRSuU81PU?t=831`

attention is all you need

from the architecture perspective, the transformer is a stack of encoders and decoders.
gpt-2 only has the decoder part, and it's a stack of decoders.

2 main differences
we do a reshuffling of the layer norms

and an additional layer normalization is added after the final self-attention layer.

in the schema, you have several layer norms in yellow, and instead of being after the mlp (multi-layer perceptron) block or after the attention, they swing before it.
and and additional layer norm is added right before the final classifier.

within the dictionnary, we have the weights of
the token embeddings WT and that's an N
embedding 

and the weights of the
position embeddings which is also just
an N embedding 

and if you remember n
embedding is really just a fancy little
wrapper module

around just a single array of numbers, a single block of numbers (like the gray graph), it's a
single tensor

and an embedding is a
glorified um wrapper 

around a tensor that allows you to access its elements by indexing into the rows

if we look at the cell with state_dict

we see that we have transformer.h.0. etc
[...]
 transformer.h.1. etc

all the way to 11 because we have 12 layers in this transformer.

in the gpt class, we are going to make a module list of 12 transformer blocks instead of a module dict.
we can index it using integers, like we see in the state dict.

the classifier at then end projects the 768 dimensional output of the last layer all the way to the size of the vocabulary.

we also don't use any bias.

wte is the red bloc, the output embedding
wpe is the positional encoding, the yin yang symbol
h is the block in gray, the transformer block
ln_f is the new layer added between the gray block and the linear
lm_head is the linear layer in dark blue.


we're now gonna use recursion to define the blocks of the transformer.

in the schema, we see that the layer normalizations, in yellow, are after the application of attention and feed forward. 

Note that also, the normalization is inside the residual stream. We have feed forward, in blue, and before that  we have both an arrow that goes to it an an arrow that goes around it. It goes through the normalization. That means that the residual pathway has normalizations inside them.

This is not that good, we want to have a single clean visual stream. from supervision, all the way down to the inputs, the tokens.

The gradients that float from the top 

if you remember from your macrograds, addition just distributes gradients during the backward state of both of its branches, equally.

The gradients from the top flow straight to the inputs, the tokens, through the residual pathway, unchanged. 

in addition to that, the gradient also flows through the blocks. 

the blocks give their own contribution over time and kick in, and change the optimization overtime.

in short, clean visual pathways are good from an optimization perspective.

our x, in the recursion call of 

x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))

x goes through the layer normalization, and then through the attention, and then through the layer normalization again 2, and the mlp

it's called a feed forward network, or FFN

finally it goes into the residual stream again.

attention is a communication operation. 

You have your 1024 tokens lined up in the sequence, and it is in the attention phase that they exchange information.

attention is an aggregation operation

attention is a pooling function

attention is a weighted sum function

attention is a reduce operation

mlp by comparison, is apllied to each token independently, there is no information being collected or exchanged between the tokens.

attention is the reduce, mlp is the map.

in short, transformers are a mix of map and reduce operations.

ok stopping at this 

`https://youtu.be/l8pRSuU81PU?t=1235`

now we do the MLP class. We could have used the approximation of tanh, but instead we use the gelu function.

by the way tanh is the hyperbolic tangent function, and gelu is the gaussian error linear unit. tanh is expressd by the hyperbolic of the exponential of x minus the exponential of minus x, divided by the exponential of x plus the exponential of minus x. gelu is expressed by x times the sigmoid of 1.702 times x.

tanh(x) = sinh(x) / cosh(x) Alternatively, it can be defined using exponential functions: tanh(x) = (e^x - e^-x) / (e^x + e^-x)

gelu(x) = x * sigmoid(1.702 * x)

there was a paper that said that gelu was better of using the approximate version, because back then the approximate version was faster. Now however, we can use the exact version.

we use gelu instead of relu because of the dead neuron problem

whereby in the tail of the relu function, the gradient is 0. 

because any activations that follow the tale will get exactly 0 gradient.


there is no change, no adatpation, no development of the network. 

Gelu, however, always contribute a local gradient, even if it's small.

There will be a smoothing it out ends up empirically working better in practice, as demonstrated in this paper

and also as demonstrated by it being picked up by the birt????

https://youtu.be/l8pRSuU81PU?t=1411
 
papers, gpt2 paper and so on

so for that reason we adopt this nonlinearity here in the tanh, in the gpt2 reproduction. 

In more modern networks also like llama 3 and so on,  this nonlinearity also further changes uh to  ??? and other variants like that but for gpt2 they 

## finally we have the attention operation

this is not just attention, this is multi-headed attention

in parallel inside every attention block there's multiple heads and they're all functioning in parallel

their outputs are just being concatenated and that becomes the output of the multi-headed attention

the heads are just kind of like parallel streams and their outputs get concatenated 

it made the head straightforward in terms of its implementation 

instead of having two separate modules or even more modules that get concatenated 

all of that is just put into a single self attention module we are doing a lot of transpose and split tensor gymnastics to make this approach very efficient in pytorch

algorithmically nothing is different from the implementation we saw before

we have these tokens lined up in a sequence and there's 1020 of them

then each token at this stage of the attention emits three vectors

the query,  the key and the value 

the queries and the keys have to multiply each other to get attention amount

or quantifying how interesting the tokens find each other

interact multiplicatively

we're calculating the qkv

we splitting it 

and we are making the number of heads and H into a batch Dimension and so it's a batch Dimension

it is just like B

in the operations that are below 

pytorch treats B and NH as batches and it applies all the operations on all of them in parallel

in both the batch and the heads

the operations that get applied are number one

the queries and the keys interact to give us their attention

the autoregressive mask makes sure that the tokens only attend to tokens before them 

and never to tokens in the future 

the softmax normalizes the attention so it sums to one always

doing the attention Matrix multiply with the values

is a weighted sum of the values of the tokens

that we found interesting at every single token

the .transpose, .contiguous, .view,

is just reassembling all of that again

it performs the concatenation operation

it is the equivalent mathematically to our previous implementation

it is just more efficient in pytorch

for the variable names, c_attn in train_gpt2.py is the same as what we see in the output cell in  cell 2 of play.ipynb

transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])

our keys should exactly follow the schema of the hugging face transformer's code

it will make it easy for us to port over all

the weights from this naming convention because all of our

variables are named the same thing

since the implementation is almost finished, we do not need to use the huggincgface transformers/src/transformers/models/gpt2/modeling_gpt2.py

this other file is 2000 lines of code, and our current code is less than 100 lines

at this stage we should just be able to take over all the weights

set them and then do generation 

let's see what that looks like

https://youtu.be/l8pRSuU81PU?t=1705