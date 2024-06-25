import torch
import torch.nn as nn
from transformers import RobertaModel

from tell.models.resnet import resnet152
from tell.modules.criteria import AdaptiveLoss
from tell.modules.token_embedders import AdaptiveEmbedding, SinusoidalPositionalEmbedding
from tell.modules.attention import MultiHeadAttention
from tell.modules.convolutions import DynamicConv1dTBC


class OwnDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Load pre-trained models
        self.roberta = RobertaModel.from_pretrained(config.model_name)
        self.resnet = resnet152()

        # Freeze ResNet and RoBERTa weights
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Custom decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.decoder_layers)
        ])

        # Other components
        self.embedding = SumEmbedding(config)
        self.adaptive_softmax = AdaptiveLoss(config)

        if config.weigh_bert:
            self.bert_weight = nn.Parameter(torch.Tensor(config.bert_layers))
            nn.init.uniform_(self.bert_weight)

    def forward(self, batch):
        # Process image
        image_features = self.resnet(batch['image'])
        image_features = image_features.view(image_features.size(0), -1, image_features.size(-1))

        # Process article
        article_outputs = self.roberta(batch['context']['roberta'],
                                       attention_mask=batch['context']['roberta'] != self.config.padding_value)

        if self.config.weigh_bert:
            article_features = torch.stack(article_outputs.hidden_states, dim=2)
            weight = torch.softmax(self.bert_weight, dim=0).unsqueeze(0).unsqueeze(1).unsqueeze(3)
            article_features = (article_features * weight).sum(dim=2)
        else:
            article_features = article_outputs.last_hidden_state

        # Process caption
        caption_embeddings = self.embedding(batch['caption'])

        # Combine features and pass through decoder layers
        decoder_input = torch.cat([image_features, article_features, caption_embeddings], dim=1)
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input)

        # Generate output
        output = self.adaptive_softmax(decoder_input)

        return {
            'logits': output,
            'loss': self.compute_loss(output, batch['target'])
        }

    def compute_loss(self, logits, targets):
        return self.adaptive_softmax(logits, targets)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config.hidden_size, config.decoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_attn = MultiHeadAttention(config.hidden_size, config.decoder_attention_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_embed_dim, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.conv = DynamicConv1dTBC(config.hidden_size, config.hidden_size, kernel_size=3)

    def forward(self, x):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x, key=x, value=x)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(query=x, key=x, value=x)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.conv(x.transpose(0, 1)).transpose(0, 1)
        x = residual + x

        return x


class SumEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adaptive = AdaptiveEmbedding(config.vocab_size, config.hidden_size)
        self.position = SinusoidalPositionalEmbedding(config.hidden_size)

    def forward(self, tokens):
        return self.adaptive(tokens) + self.position(tokens)
