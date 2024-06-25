import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from decoder_faces_objects import (DynamicConvDecoderConfig, DynamicConvDecoder)
from dataset import NYTimesDataset
from model import TransformAndTellModel

# Create an instance of the DynamicConvDecoder model
config = DynamicConvDecoderConfig(
    vocab_size=30000,
    max_position_embeddings=512,
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_target_positions=512,
    dropout=0.1,
    share_decoder_input_output_embed=True,
    decoder_output_dim=768,
    decoder_conv_dim=768,
    decoder_glu=True,
    decoder_conv_type='dynamic',
    weight_softmax=True,
    weight_dropout=0.1,
    relu_dropout=0.1,
    input_dropout=0.1,
    decoder_normalize_before=False,
    decoder_kernel_size_list=[3, 7, 15, 31, 63, 127],
    adaptive_softmax_cutoff=None,
    tie_adaptive_weights=False,
    adaptive_softmax_dropout=0.0,
    tie_adaptive_proj=False,
    adaptive_softmax_factor=4,
    decoder_layers=6,
    final_norm=False,
    padding_idx=1,
    namespace='target_tokens',
    section_attn=False,
    article_embed_size=512
)

model = DynamicConvDecoder(config)

# Prepare the data
json_dir = './sample/sample_json'
image_dir = './sample/sample_images'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_dataset = NYTimesDataset(json_dir, image_dir, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Create an instance of the TransformAndTellModel
model = TransformAndTellModel(vocab_size=tokenizer.vocab_size)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        caption_input_ids = batch['caption_input_ids'].to(device)
        caption_attention_mask = batch['caption_attention_mask'].to(device)
        article_input_ids = batch['article_input_ids'].to(device)
        article_attention_mask = batch['article_attention_mask'].to(device)
        image = batch['image'].to(device)
        face_embeddings = batch['face_embeddings'].to(device)
        object_embeddings = batch['object_embeddings'].to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(article_input_ids, article_attention_mask, image, face_embeddings, object_embeddings, caption_input_ids)

        # Compute loss
        loss = criterion(output.view(-1, tokenizer.vocab_size), caption_input_ids.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'transform_and_tell_model.pth')
