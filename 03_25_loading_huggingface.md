https://youtu.be/l8pRSuU81PU?t=1705


leaves of the BP tree and one special end of text token

that delimits different documents and can start generation as well

load the parameters from hugging face to code here and initialize the GPT class

with those parameters 

we're just loading the weights so it's dry 

there are four models in this miniseries of gpt2

config args are the hyper parameters of the gpt2 model

we're creating the config object and creating our own model

then we're creating the state dict both for our model and for the hugging face model

we're going over the hugging face model keys

we're copying over those tensors 

in the process we ignore a few of the buffers

They are not parameters they're buffers

for example .attn.bias

it is used for the autoaggressive mask

we are ignoring some of those masks

this comes from the tensorflow repo

some of the weights are transposed from what pytorch would want

we had to hardcode the weights

that should be transposed and if they appear then we transpose them

we return this model

from_pretrained is a class method in Python

that Returns the GPT object if we give it the model type 

in our case it is gpt2 the smallest model

we can python train_gpt2.py

it didn't crash, so we can load the weights and the biases

let us generate from this module

https://youtu.be/l8pRSuU81PU?t=1856