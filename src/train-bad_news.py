import torch
import torch.nn as nn

from decoder import TransformerDecoder, TransformerDecoderLayer
from model import BadNews

d_model = 512
vocab_size = 10000
num_contexts = 2
num_heads = 4


tgt_seq_length = 50
# tgt = torch.randint(0, vocab_size, (tgt_seq_length, 1))

tgt = torch.rand(tgt_seq_length, 32, d_model)

context_lengths = [60, 1, 5, 7]
context_dims = [512, 1024, 512, 1024]
contexts = [torch.rand(context_lengths[idx], 32, d_model) for idx in range(num_contexts)]
print(f"Shape of first context: {contexts[0].shape}")
print(f"ncontexts: {num_contexts}")

decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads)
transformer = nn.TransformerDecoder(decoder, 2)
out = transformer.forward(tgt, tgt)
print(f"Torch Model Output: {out.shape}")

decoder_self = TransformerDecoderLayer(d_model=d_model, nhead=num_heads, ncontexts=num_contexts)
transformer_self = TransformerDecoder(decoder_self, 2)
out_self = decoder_self.forward(tgt, contexts)
print(f"Own Model Output: {out_self.shape}")

model = BadNews(
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=num_heads,
    num_decoder_layers=2,
    dim_feedforward=2048,
    max_seq_length=512,
    ncontexts=num_contexts,
)

tgt_seq_length = 50
tgt = torch.randint(0, vocab_size, (tgt_seq_length, 1))

# model.forward(tgt, contexts)

# Forward Pass:
# Positional Embedding -> Self-Attention -> Norm -> Cross-Attention (Self + Contexts) for multiple contexts -> Norm -> Feed Foward -> Norm -> Repeat n times -> Linear Layer
