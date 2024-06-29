import torch
import torch.nn as nn

decoder = nn.TransformerDecoderLayer(512, 1)
encoder = nn.TransformerEncoderLayer(512, 1)
