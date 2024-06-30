import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, CenterCrop
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)


class TaTDatasetReader(Dataset):
    def __init__(
            self,
            image_encoder,
            image_dir: str,
            mongo_host: str = 'localhost',
            mongo_port: int = 27017,
            use_caption_names: bool = True,
            use_objects: bool = False,
            n_faces: int = None,
            roberta_model: str = 'roberta-base',
            max_length: int = 512,
            context_before: int = 8,
            context_after: int = 8,
            seed: int = 464896,
            device: str = 'cuda',
            split: str = 'train',
            dtype: torch.dtype = torch.float32,
    ):
        self.client = MongoClient(f"mongodb://root:secure_pw@{mongo_host}:{mongo_port}/")
        self.db = self.client.nytimes
        self.context_before = context_before
        self.context_after = context_after

        self.preprocess = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.use_caption_names = use_caption_names
        self.use_objects = use_objects
        self.n_faces = n_faces
        self.max_length = max_length
        self.rs = np.random.RandomState(seed)
        self.device = device
        self.dtype = dtype

        self.image_dir = image_dir
        self.image_encoder = image_encoder
        self.image_transforms = Compose([
            Resize((256, 256)),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.roberta.to(device)
        self.roberta.eval()

        # self.data = self._load_data(split)

    def load_data(self, split: str):
        """Load and preprocess the dataset."""
        logger.info(f'Loading {split} data')
        data = []
        projection = ['_id', 'parsed_section', 'image_positions', 'headline', 'web_url']

        for article in tqdm(self.db.articles.find({'split': split}, projection=projection)):
            article_data = self._process_article(article)
            data.extend(article_data)

        return data

    def _process_article(self, article):
        """Process a single article and all its images."""
        article_data = []
        sections = article['parsed_section']

        for pos in article['image_positions']:
            if pos >= len(sections) or sections[pos]['type'] != 'caption':
                continue

            caption = sections[pos]['text'].strip()
            if not caption:
                continue

            image_hash = sections[pos]['hash']
            image_path = os.path.join(self.image_dir, f"{image_hash}.jpg")
            try:
                image = Image.open(image_path).convert('RGB')
            except (FileNotFoundError, OSError):
                logger.error(f"Could not open image at {image_path}")
                continue

            # Caption
            tokenized_caption = self.tokenizer(caption, max_length=self.max_length, truncation=True, padding='max_length')
            caption_input_ids = tokenized_caption["input_ids"]
            caption_attention_mask = tokenized_caption["attention_mask"]
            outputs = self.roberta(caption_input_ids, attention_mask=caption_attention_mask)
            caption_embedding = outputs.last_hidden_state.transpose(0, 1)

            # Text
            text = self._get_context(article, pos)
            tokenized_text = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length')
            text_input_ids = tokenized_text["input_ids"]
            text_attention_mask = tokenized_text["attention_mask"]
            outputs = self.roberta(text_input_ids, attention_mask=text_attention_mask)
            text_embedding = outputs.last_hidden_state.transpose(0, 1)

            # Image
            image_tensor = self.image_transforms(image).unsqueeze(0)
            with torch.no_grad():
                image_embedding = self.image_transforms(image_tensor)
                image_embedding = self.image_encoder(image_embedding)

            # Faces
            if 'facenet_details' in sections[pos]:
                faces = sections[pos]['facesnet_details']['embeddings']
                tmp = torch.tensor(faces).unsqueeze(1)
                faces = torch.zeros(faces.shape[0], 1, self.tokenizer.hidden_size)
                faces[:,:,:512] = tmp
            else:
                faces = torch.zeros(0, 1, self.tokenizer.config.hidden_size)
            image_embedding = image_embedding.to(self.device, dtype=self.dtype)

            # Objects
            objects = self.db.objects.find_one({'_id': image_hash})
            if objects is not None:
                objects = objects['objects_features']
            else:
                objects = torch.zeros(0, 1, self.tokenizer.config.hidden_size)

            article_data.append({
                'image': image_embedding,
                'caption': caption_embedding,
                'context': text_embedding,
                'faces': faces,
                'objects': objects,
            })

        return article_data

    def _get_context(self, article, pos):
        """Extract context from the article."""
        context = []
        if 'main' in article['headline']:
            context.append(article['headline']['main'].strip())

        sections = article['parsed_section']
        for i, section in enumerate(sections):
            if section['type'] == 'paragraph':
                if pos - self.context_before <= i <= pos + self.context_after:
                    context.append(section['text'])

        return ' '.join(context)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = self.image_transform(item['image'])

        context_encoding = self.tokenizer.encode_plus(
            item['context'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        caption_encoding = self.tokenizer.encode_plus(
            item['caption'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            context_embeddings = self.roberta(
                context_encoding['input_ids'].squeeze(0),
                attention_mask=context_encoding['attention_mask'].squeeze(0)
            )[0]

        return {
            'image': image,
            'context_ids': context_encoding['input_ids'].squeeze(0),
            'context_mask': context_encoding['attention_mask'].squeeze(0),
            'context_embeddings': context_embeddings,
            'caption_ids': caption_encoding['input_ids'].squeeze(0),
            'caption_mask': caption_encoding['attention_mask'].squeeze(0),
            'named_entities': item['named_entities'],
        }
