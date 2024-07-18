from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, MultiheadAttention

from decoder_utils import (detect_is_causal_mask, get_activation_fn,
                           get_clones, get_seq_len)


# Decoder layer based on the pytorch implementation.


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = False,
            norm_first: bool = False,
            bias: bool = True,
            device=None,
            dtype=None,
            ncontexts: int = 1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.ncontexts = ncontexts

        # create one multi head attention for each context
        self.context_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d_model,
                    nhead,
                    dropout=dropout,
                    batch_first=batch_first,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(ncontexts)
            ],
        )

        # Implementation of Feedforward model
        # adjust linear layer to handle a concatenated tensor from multiple context attentions
        self.linear1 = Linear(d_model * ncontexts, dim_feedforward, bias=bias, device=device, dtype=dtype)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, device=device, dtype=dtype)

        self.norm_first = norm_first

        # norm for self attention
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
        # norm for multi head attentions for each context
        self.context_norms = nn.ModuleList(
            [LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype) for _ in range(ncontexts)],
        )
        # norm for ff
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
        # dropout for self attention
        self.dropout1 = Dropout(dropout)
        # dropout for multi head attentions for each context
        self.context_drop_outs = nn.ModuleList([Dropout(dropout) for _ in range(ncontexts)])
        # dropout for ff
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(
            self,
            tgt: Tensor,
            memory: List[Tensor],
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
    ) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        # ensure that number of contexts is correct
        assert len(memory) == self.ncontexts, "Number of contexts do not match the memory dimension 0"
        x = tgt
        sa_input = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
        attentions = []
        # attend to each context
        for idx, context in enumerate(memory):
            attentions.append(
                self.context_norms[idx](
                    sa_input
                    + self._mha_block(sa_input, context, memory_mask, memory_key_padding_mask, memory_is_causal, idx)
                )
            )
        # concatenate attention array
        concat_attention = torch.concat(attentions, dim=-1)
        x = self.norm3(x + self._ff_block(concat_attention))
        return x

    # self-attention block
    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
            self,
            x: Tensor,
            mem: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            is_causal: bool = False,
            index: int = 0,
    ) -> Tensor:
        x = self.context_attentions[index](
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.context_drop_outs[index](x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


# --- decoder ---


class TransformerDecoder(nn.Module):
    """TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    __constants__ = ["norm"]

    def __init__(
            self,
            decoder_layer: "TransformerDecoderLayer",
            num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(
            self,
            tgt: Tensor,
            memory: List[Tensor],  # this has to be a list of tensors
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: Optional[bool] = None,
            memory_is_causal: bool = False,
    ) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        output = tgt

        seq_len = get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                # give whole context to each decoder layer
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        return output
