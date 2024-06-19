import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class NYTimesDataset(Dataset):
    def __init__(self, json_dir, image_dir, tokenizer, max_seq_length=512, max_image_size=512):
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_image_size = max_image_size
        self.json_files = [file for file in os.listdir(json_dir) if file.endswith('.json')]

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = os.path.join(self.json_dir, self.json_files[idx])
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract relevant fields from the JSON data
        headline = data['headline']['main']
        article = ' '.join([section['text'] for section in data['parsed_section']])
        image_hashes = [image['hash'] for image in data['multimedia'] if image['type'] == 'image']

        # Tokenize headline and article
        headline_inputs = self.tokenizer(headline, add_special_tokens=True, max_length=self.max_seq_length,
                                         padding='max_length', truncation=True, return_tensors='pt')
        article_inputs = self.tokenizer(article, add_special_tokens=True, max_length=self.max_seq_length,
                                        padding='max_length', truncation=True, return_tensors='pt')

        # Load and preprocess images
        images = []
        for hash in image_hashes:
            image_path = os.path.join(self.image_dir, hash)
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.max_image_size, self.max_image_size))
            image = torch.tensor(image).permute(2, 0, 1)  # Convert to torch tensor and transpose to (C, H, W)
            images.append(image)

        if len(images) == 0:
            images = torch.zeros(1, 3, self.max_image_size, self.max_image_size)
        else:
            images = torch.stack(images)

        # Extract face and object embeddings
        face_embeddings = []
        object_embeddings = []
        for section in data['parsed_section']:
            if 'facenet_details' in section:
                face_embeddings.extend(section['facenet_details']['embeddings'])
            if 'object_details' in section:
                object_embeddings.extend(section['object_details']['embeddings'])

        if len(face_embeddings) == 0:
            face_embeddings = torch.zeros(1, 512)  # Assuming face embeddings have 512 dimensions
        else:
            face_embeddings = torch.tensor(face_embeddings)

        if len(object_embeddings) == 0:
            object_embeddings = torch.zeros(1, 512)  # Assuming object embeddings have 512 dimensions
        else:
            object_embeddings = torch.tensor(object_embeddings)

        return {
            'headline_input_ids': headline_inputs['input_ids'].squeeze(),
            'headline_attention_mask': headline_inputs['attention_mask'].squeeze(),
            'article_input_ids': article_inputs['input_ids'].squeeze(),
            'article_attention_mask': article_inputs['attention_mask'].squeeze(),
            'images': images,
            'face_embeddings': face_embeddings,
            'object_embeddings': object_embeddings
        }


# Create an instance of the NYTimesDataset
json_dir = '/home/marlon/Git/sample/sample_json'
image_dir = '/home/marlon/Git/sample/sample_images'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = NYTimesDataset(json_dir, image_dir, tokenizer)

# Create a DataLoader
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
