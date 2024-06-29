import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bert.modeling_bert import BertEmbeddings

from modules import (AdaptiveSoftmax, DynamicConv1dTBC, GehringLinear,
                     LightweightConv1dTBC, MultiHeadAttention)


class DynamicConvDecoderConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size,
        max_position_embeddings,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        max_target_positions,
        dropout,
        share_decoder_input_output_embed,
        decoder_output_dim,
        decoder_conv_dim,
        decoder_glu,
        decoder_conv_type,
        weight_softmax,
        weight_dropout,
        relu_dropout,
        input_dropout,
        decoder_normalize_before,
        decoder_kernel_size_list,
        adaptive_softmax_cutoff,
        tie_adaptive_weights,
        adaptive_softmax_dropout,
        tie_adaptive_proj,
        adaptive_softmax_factor,
        decoder_layers,
        final_norm,
        padding_idx,
        namespace,
        section_attn,
        article_embed_size,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_target_positions = max_target_positions
        self.dropout = dropout
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.decoder_output_dim = decoder_output_dim
        self.decoder_conv_dim = decoder_conv_dim
        self.decoder_glu = decoder_glu
        self.decoder_conv_type = decoder_conv_type
        self.weight_softmax = weight_softmax
        self.weight_dropout = weight_dropout
        self.relu_dropout = relu_dropout
        self.input_dropout = input_dropout
        self.decoder_normalize_before = decoder_normalize_before
        self.decoder_kernel_size_list = decoder_kernel_size_list
        self.adaptive_softmax_cutoff = adaptive_softmax_cutoff
        self.tie_adaptive_weights = tie_adaptive_weights
        self.adaptive_softmax_dropout = adaptive_softmax_dropout
        self.tie_adaptive_proj = tie_adaptive_proj
        self.adaptive_softmax_factor = adaptive_softmax_factor
        self.decoder_layers = decoder_layers
        self.final_norm = final_norm
        self.padding_idx = padding_idx
        self.namespace = namespace
        self.section_attn = section_attn
        self.article_embed_size = article_embed_size


class DynamicConvDecoder(nn.Module):
    config_class = DynamicConvDecoderConfig

    def __init__(self, config):
        self.embeddings = BertEmbeddings(config)  # todo p4 make generic
        self.layers = nn.ModuleList(
            [
                DynamicConvDecoderLayer(config, kernel_size=config.decoder_kernel_size_list[i])
                for i in range(config.decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size) if config.final_norm else None
        self.dropout = nn.Dropout(config.dropout)

        self.adaptive_softmax = None

        embed_dim = config.hidden_size
        output_embed_dim = config.hidden_size
        super().__init__(config)

        def eval_str_list(x, type=float):
            if x is None:
                return None
            if isinstance(x, str):
                x = eval(x)
            try:
                return list(map(type, x))
            except TypeError:
                return [type(x)]

        if config.adaptive_softmax_cutoff is not None:
            adaptive_inputs = None
            if isinstance(self.embeddings, BertEmbeddings):  # todo likely bug here
                adaptive_inputs = self.embeddings
            elif hasattr(self.embeddings, "token_embedder_adaptive"):
                adaptive_inputs = self.embeddings.token_embedder_adaptive
            elif config.tie_adaptive_weights:
                raise ValueError("Cannot locate adaptive_inputs.")
            self.adaptive_softmax = AdaptiveSoftmax(
                config.vocab_size,
                output_embed_dim,
                eval_str_list(config.adaptive_softmax_cutoff, type=int),
                dropout=config.adaptive_softmax_dropout,
                adaptive_inputs=adaptive_inputs,
                factor=config.adaptive_softmax_factor,
                tie_proj=config.tie_adaptive_proj,
            )
        elif not config.share_decoder_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(config.vocab_size, output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim**-0.5)

        self.register_buffer("version", torch.tensor([2], dtype=torch.float32))
        self.normalize = config.decoder_normalize_before and config.final_norm
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, prev_target, contexts, incremental_state=None, use_layers=None, **kwargs):
        X = self.embeddings(prev_target)
        X = self.dropout(X)
        X = X.transpose(0, 1)
        attn = None

        inner_states = [X]
        for i, layer in enumerate(self.layers):
            if not use_layers or i in use_layers:
                X, attn = layer(X, contexts, incremental_state)
                inner_states.append(X)

        if self.layer_norm is not None:
            X = self.layer_norm(X)

        X = X.transpose(0, 1)

        return BaseModelOutput(last_hidden_state=X, hidden_states=inner_states, attentions=attn)


class DynamicConvDecoderLayer(nn.Module):
    def __init__(self, config, kernel_size):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.conv_dim = config.decoder_conv_dim
        if config.decoder_glu:
            self.linear1 = GehringLinear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = GehringLinear(self.embed_dim, self.conv_dim)
            self.act = None

        if config.decoder_conv_type == "lightweight":
            self.conv = LightweightConv1dTBC(
                self.conv_dim,
                kernel_size,
                padding_l=kernel_size - 1,
                weight_softmax=config.weight_softmax,
                num_heads=config.num_attention_heads,
                weight_dropout=config.weight_dropout,
            )
        elif config.decoder_conv_type == "dynamic":
            self.conv = DynamicConv1dTBC(
                self.conv_dim,
                kernel_size,
                padding_l=kernel_size - 1,
                weight_softmax=config.weight_softmax,
                num_heads=config.num_attention_heads,
                weight_dropout=config.weight_dropout,
            )
        else:
            raise NotImplementedError

        self.linear2 = GehringLinear(self.conv_dim, self.embed_dim)

        self.dropout = config.dropout
        self.relu_dropout = config.relu_dropout
        self.input_dropout = config.input_dropout
        self.normalize_before = config.decoder_normalize_before

        self.conv_layer_norm = nn.LayerNorm(self.embed_dim)

        self.context_attns = nn.ModuleDict()
        self.context_attn_lns = nn.ModuleDict()

        # todo make generic
        self.context_attns["image"] = MultiHeadAttention(
            self.embed_dim,
            config.num_attention_heads,
            kdim=2048,
            vdim=2048,
            dropout=config.attention_probs_dropout_prob,
        )
        self.context_attn_lns["image"] = nn.LayerNorm(self.embed_dim)

        self.context_attns["article"] = MultiHeadAttention(
            self.embed_dim,
            config.num_attention_heads,
            kdim=config.article_embed_size,
            vdim=config.article_embed_size,
            dropout=config.attention_probs_dropout_prob,
        )
        self.context_attn_lns["article"] = nn.LayerNorm(self.embed_dim)

        self.context_attns["faces"] = MultiHeadAttention(
            self.embed_dim, config.num_attention_heads, kdim=512, vdim=512, dropout=config.attention_probs_dropout_prob
        )
        self.context_attn_lns["faces"] = nn.LayerNorm(self.embed_dim)

        self.context_attns["obj"] = MultiHeadAttention(
            self.embed_dim,
            config.num_attention_heads,
            kdim=2048,
            vdim=2048,
            dropout=config.attention_probs_dropout_prob,
        )
        self.context_attn_lns["obj"] = nn.LayerNorm(self.embed_dim)

        context_size = self.embed_dim * 4  # maybe 2

        self.context_fc = GehringLinear(context_size, self.embed_dim)

        self.fc1 = GehringLinear(self.embed_dim, config.intermediate_size)
        self.fc2 = GehringLinear(config.intermediate_size, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self, X, contexts, incremental_state):
        residual = X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, before=True)
        X = F.dropout(X, p=self.input_dropout, training=self.training)
        X = self.linear1(X)
        if self.act is not None:
            X = self.act(X)
        X = self.conv(X, incremental_state=incremental_state)
        X = self.linear2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, after=True)

        attns = {}
        X_contexts = []

        for context_type in ["image", "article", "faces", "obj"]:
            residual = X
            X_context = self.maybe_layer_norm(self.context_attn_lns[context_type], X, before=True)
            X_context, attn = self.context_attns[context_type](
                query=X_context,
                key=contexts[context_type],
                value=contexts[context_type],
                key_padding_mask=contexts[context_type + "_mask"],
                incremental_state=None,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            X_context = F.dropout(X_context, p=self.dropout, training=self.training)
            X_context = residual + X_context
            X_context = self.maybe_layer_norm(self.context_attn_lns[context_type], X_context, after=True)
            X_contexts.append(X_context)
            if attn is not None:
                attns[context_type] = attn.detach().cpu().numpy()

        X_context = torch.cat(X_contexts, dim=-1)
        X = self.context_fc(X_context)

        residual = X
        X = self.maybe_layer_norm(self.final_layer_norm, X, before=True)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.relu_dropout, training=self.training)
        X = self.fc2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.final_layer_norm, X, after=True)

        return X, attns

    def maybe_layer_norm(self, layer_norm, X, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(X)
        else:
            return X

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def extra_repr(self):
        return f"dropout={self.dropout}, relu_dropout={self.relu_dropout}, input_dropout={self.input_dropout}, normalize_before={self.normalize_before}"
