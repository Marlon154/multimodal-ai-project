import torch
import torch.nn as nn

from decoder import TransformerDecoder, TransformerDecoderLayer


class BadNews(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        ncontexts,
        device,
    ):
        super(BadNews, self).__init__()

        self.device = device
        self.positional_encoding = self._generate_positional_encoding(d_model, max_seq_length)

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            ncontexts=ncontexts,
            batch_first=True,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self.linear = nn.Linear(d_model, vocab_size)

    def _generate_positional_encoding(self, d_model, max_seq_length):
        positional_encoding = torch.zeros(max_seq_length, d_model).to(self.device)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        return positional_encoding

    def forward(self, tgt_embeddings, contexts):
        tgt_embeddings += self.positional_encoding[: tgt_embeddings.size(0), :]

        # context adapter
        decoded_output = self.decoder(tgt=tgt_embeddings, memory=contexts)

        output = self.linear(decoded_output)

        return output
