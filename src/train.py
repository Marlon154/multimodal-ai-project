import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.decoder_faces_objects import DynamicConvDecoderConfig, DynamicConvDecoder
from src.dataset import NYTimesDataset

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
    decoder_kernel_size_list=[3, 7, 15],
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
image_dir = './sample_images'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = NYTimesDataset(json_dir, image_dir, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        headline_input_ids = batch['headline_input_ids']
        headline_attention_mask = batch['headline_attention_mask']
        article_input_ids = batch['article_input_ids']
        article_attention_mask = batch['article_attention_mask']
        images = batch['images']
        face_embeddings = batch['face_embeddings']
        object_embeddings = batch['object_embeddings']

        optimizer.zero_grad()

        # Forward pass
        output = model(prev_target=article_input_ids, contexts={
            'image': images,
            'article': article_input_ids,
            'faces': face_embeddings,
            'obj': object_embeddings,
            'image_mask': None,
            'article_mask': article_attention_mask,
            'faces_mask': None,
            'obj_mask': None
        })

        # Compute loss
        loss = criterion(output.last_hidden_state.view(-1, config.vocab_size), article_input_ids.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'dynamic_conv_decoder.pth')
