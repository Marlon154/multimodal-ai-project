import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiContextAttention(nn.Module):
    def __init__(self, d_model, nhead, num_contexts):
        super(MultiContextAttention, self).__init__()
        self.num_contexts = num_contexts
        self.d_model = d_model
        self.nhead = nhead
        self.attn_heads = nn.ModuleList(
            [nn.MultiheadAttention(d_model, nhead) for _ in range(num_contexts)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model * num_contexts, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()

    def forward(self, tgt, contexts, tgt_mask=None, tgt_key_padding_mask=None):
        context_outputs = []

        # Calculate attention for each context
        for i, context in enumerate(contexts):
            context_output, _ = self.attn_heads[i](
                tgt,
                context,
                context,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
            context_outputs.append(context_output)

        # Concatenate context outputs
        concat_contexts = torch.cat(context_outputs, dim=-1)

        # Feed through feedforward layers
        output = self.fc1(concat_contexts)
        output = self.fc2(output)
        output = self.relu(output)

        # Add residual connection and layer normalization
        output = self.norm(tgt + output)

        return output


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        num_contexts,
    ):
        super(DecoderOnlyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(
            d_model, max_seq_length
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.multi_context_attention = MultiContextAttention(
            d_model, nhead, num_contexts
        )
        self.linear = nn.Linear(d_model, vocab_size)

    def _generate_positional_encoding(self, d_model, max_seq_length):
        positional_encoding = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        return positional_encoding

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, tgt, contexts):
        tgt_embedded = self.embedding(tgt) * torch.sqrt(
            torch.tensor(self.embedding.embedding_dim, dtype=torch.float)
        )
        tgt_embedded += self.positional_encoding[: tgt.size(0), :]

        tgt_seq_length = tgt.size(0)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_length)

        # Apply decoder
        decoded_output = self.decoder(tgt_embedded, memory=None, tgt_mask=tgt_mask)

        # Apply multi-context attention
        attended_output = self.multi_context_attention(
            decoded_output, contexts, tgt_mask=tgt_mask
        )

        # Final linear layer
        output = self.linear(attended_output)

        return output


vocab_size = 10000
d_model = 512
nhead = 8
num_decoder_layers = 6
dim_feedforward = 2048
max_seq_length = 100
num_contexts = 4

model = DecoderOnlyTransformer(
    vocab_size,
    d_model,
    nhead,
    num_decoder_layers,
    dim_feedforward,
    max_seq_length,
    num_contexts,
)


# Example target sequence (batch_size, tgt_seq_length)
tgt_seq_length = 50
tgt = torch.randint(0, vocab_size, (tgt_seq_length, 1))
print(tgt)

# Dummy contexts (each context has shape [context_length, batch_size, d_model])
context_length = 60
contexts = [torch.rand(context_length, 32, d_model) for _ in range(num_contexts)]
print(contexts[0].shape)

# Forward pass
output = model(tgt, contexts)
