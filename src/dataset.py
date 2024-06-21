import os
import json

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class NYTimesDataset(Dataset):
    def __init__(self, json_dir, image_dir, tokenizer, max_seq_length=512, max_image_size=512, max_images=4, transform=None):
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_image_size = max_image_size
        self.json_files = [file for file in os.listdir(json_dir) if file.endswith('.json')]
        self.max_images = max_images
        self.transform = transform

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = os.path.join(self.json_dir, self.json_files[idx])
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract relevant fields from the JSON data
        headline = data['headline']['main']
        article = ' '.join([section['text'] for section in data['parsed_section'] if section['type'] == 'paragraph'])

        # Iterate over sections until the one caption is found
        for section in data['parsed_section']:
            if section['type'] == 'caption':
                caption = section['text']
                if 'hash' in section:
                    image_hash = section['hash']
                else:
                    continue
                if 'facenet_details' in section:
                    face_embeddings = section['facenet_details']['embeddings'][0:self.max_images]
                else:
                    face_embeddings = []

                # Object_details does not occure in Sample
                if 'object_details' in section:
                    object_embeddings = section['object_details']['embeddings']
                else:
                    object_embeddings = []
                break
        else:
            # Skip this sample if no suitable caption is found
            return self.__getitem__((idx + 1) % len(self))

        # Tokenize headline, caption, and article
        headline_inputs = self.tokenizer(headline, add_special_tokens=True, max_length=self.max_seq_length,
                                         padding='max_length', truncation=True, return_tensors='pt')
        caption_inputs = self.tokenizer(caption, add_special_tokens=True, max_length=self.max_seq_length,
                                        padding='max_length', truncation=True, return_tensors='pt')
        article_inputs = self.tokenizer(article, add_special_tokens=True, max_length=self.max_seq_length,
                                        padding='max_length', truncation=True, return_tensors='pt')

        # Load and preprocess image
        image_path = os.path.join(self.image_dir, image_hash)
        image = Image.open(image_path + ".jpg").convert('RGB')
        image = image.resize((self.max_image_size, self.max_image_size))
        image = np.array(image)
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to torch tensor and transpose to (C, H, W)

        # Extract face and object embeddings
        if len(face_embeddings) == 0:
            face_embeddings = torch.zeros(self.max_images, 512)  # Assuming face embeddings have 512 dimensions
        else:
            face_embeddings = torch.tensor(face_embeddings)
            if len(face_embeddings) < self.max_images:
                face_embeddings = torch.cat((face_embeddings, torch.zeros(self.max_images - len(face_embeddings), 512)))

        if len(object_embeddings) == 0:
            object_embeddings = torch.zeros(1, 512)  # Assuming object embeddings have 512 dimensions
        else:
            object_embeddings = torch.tensor(object_embeddings)

        return {
            'headline_input_ids': headline_inputs['input_ids'].squeeze().to(torch.float32),
            'headline_attention_mask': headline_inputs['attention_mask'].squeeze().to(torch.float32),
            'caption_input_ids': caption_inputs['input_ids'].squeeze().to(torch.float32),
            'caption_attention_mask': caption_inputs['attention_mask'].squeeze().to(torch.float32),
            'article_input_ids': article_inputs['input_ids'].squeeze().to(torch.float32),
            'article_attention_mask': article_inputs['attention_mask'].squeeze().to(torch.float32),
            'image': image.to(torch.float32),
            'face_embeddings': face_embeddings.to(torch.float32),
            'object_embeddings': object_embeddings.to(torch.float32)
        }


# Create an instance of the NYTimesDataset
json_dir = './sample/sample_json'
image_dir = './sample/sample_images'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = NYTimesDataset(json_dir, image_dir, tokenizer)

# Create a DataLoader
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)