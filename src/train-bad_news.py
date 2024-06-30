import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

from decoder import TransformerDecoder, TransformerDecoderLayer
from model import BadNews

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
roberta_model = RobertaModel.from_pretrained("roberta-large")

text = "The cat sat on the mat"

# tokenize and encode
inputs = tokenizer(text, return_tensors="pt", max_length=10, truncation=True, padding="max_length")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print(f"Input Shape: {input_ids.shape}")

# get contextualized embeddings
with torch.no_grad():
    outputs = roberta_model(input_ids, attention_mask=attention_mask)
    tgt_embeddings = outputs.last_hidden_state.transpose(0, 1)  # Transpose to match the expected shape
    print(f"Output Shape: {tgt_embeddings.shape}")

# Create the model
d_model = 1024
num_contexts = 4
num_heads = 16
num_decoder_layers = 4
d_feedforward = 2024
vocab_size = tokenizer.vocab_size

context_lengths = [20, 5, 0, 2]
context_dims = [1024, 512, 256, 512]
contexts = [torch.rand(context_lengths[idx], 1, d_model) for idx in range(num_contexts)]
print(f"Shape of first context: {contexts[0].shape}")
print(f"ncontexts: {num_contexts}")

decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads)
out = decoder.forward(tgt_embeddings, tgt_embeddings)
print(f"Torch Model Output: {out.shape}")

decoder_self = TransformerDecoderLayer(d_model=d_model, nhead=num_heads, ncontexts=num_contexts)
out_self = decoder_self.forward(tgt_embeddings, contexts)
print(f"Own Model Output: {out_self.shape}")

# Forward Pass:
# Positional Embedding -> Self-Attention -> Norm -> Cross-Attention (Self + Contexts) for multiple contexts -> Norm -> Feed Foward -> Norm -> Repeat n times -> Linear Layer
model = BadNews(
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=num_heads,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=d_feedforward,
    max_seq_length=512,
    ncontexts=num_contexts,
)

output = model.forward(tgt_embeddings, contexts)

predicted_token_ids = torch.argmax(output, dim=-1).squeeze(1)
predicted_token_ids = predicted_token_ids.tolist()
decoded_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
print(decoded_text)
