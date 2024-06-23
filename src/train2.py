import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel

from dataset import NYTimesDataset

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, context):
        attn_output, _ = self.mha(x, context, context)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

    def forward(self, x, context):
        for layer in self.layers:
            x = layer(x, context)
        return x

class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dff, max_position_embeddings, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_position_embeddings, d_model)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, rate)
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, context):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.decoder(x, context)
        x = self.final_layer(x)
        return x

    @staticmethod
    def positional_encoding(length, d_model):
        positions = torch.arange(length).unsqueeze(1)
        divisions = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding = torch.zeros(length, d_model)
        pos_encoding[:, 0::2] = torch.sin(positions * divisions)
        pos_encoding[:, 1::2] = torch.cos(positions * divisions)
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding

class TransformAndTellModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, dff=2048, max_position_embeddings=512, rate=0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.image_encoder = nn.Linear(2048, d_model)
        self.face_encoder = nn.Linear(512, d_model)
        self.object_encoder = nn.Linear(2048, d_model)
        self.caption_generator = CaptionGenerator(vocab_size, d_model, num_layers, num_heads, dff, max_position_embeddings, rate)

    def forward(self, article_input_ids, article_attention_mask, image, face_embeddings, object_embeddings, caption_input_ids):
        # Ensure input_ids are of type Long
        article_input_ids = article_input_ids.long()
        caption_input_ids = caption_input_ids.long()

        # Encode article using RoBERTa
        article_embeddings = self.roberta(article_input_ids, attention_mask=article_attention_mask)[0]

        # Encode image
        image_embeddings = self.image_encoder(image)

        # Encode faces
        face_embeddings = self.face_encoder(face_embeddings)

        # Encode objects
        object_embeddings = self.object_encoder(object_embeddings)

        # Concatenate all embeddings
        context = torch.cat((article_embeddings, image_embeddings, face_embeddings, object_embeddings), dim=1)

        # Generate caption using transformer decoder
        caption = self.caption_generator(caption_input_ids, context)

        return caption



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
        caption_input_ids = batch['caption_input_ids'].to(device).long()
        caption_attention_mask = batch['caption_attention_mask'].to(device)
        article_input_ids = batch['article_input_ids'].to(device).long()
        article_attention_mask = batch['article_attention_mask'].to(device)
        image = batch['image'].to(device)
        face_embeddings = batch['face_embeddings'].to(device)
        object_embeddings = batch['object_embeddings'].to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(article_input_ids, article_attention_mask, image, face_embeddings, object_embeddings,
                       caption_input_ids)

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