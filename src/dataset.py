import gc
import logging
import os

import torch
from PIL import Image
from pymongo import MongoClient
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import models
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

logger = logging.getLogger(__name__)


class TaTDatasetReader(Dataset):
    def __init__(
        self,
        image_dir: str,
        mongo_host: str = "localhost",
        mongo_port: int = 27017,
        roberta_model: str = "roberta-large",
        max_length: int = 512,
        context_before: int = 8,
        context_after: int = 8,
        seed: int = 464896,
        split: str = "train",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.client = MongoClient(f"mongodb://root:secure_pw@{mongo_host}:{mongo_port}/")
        self.db = self.client.nytimes
        self.context_before = context_before
        self.context_after = context_after

        self.preprocess = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        self.data = []
        self.d_model = 1024

        self.image_dir = image_dir
        self.image_transforms = Compose(
            [
                Resize((256, 256)),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # image encoder
        resnet152 = models.resnet152(pretrained=True)
        # remove the pooling and fully connected layers
        modules = list(resnet152.children())[:-2]
        self.image_encoder = torch.nn.Sequential(*modules)
        self.image_encoder.eval()
        self.image_encoder.to(device)

        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.roberta.to(device)
        self.roberta.eval()

        # load article ids
        projection = ["_id"]
        self.article_ids = [el["_id"] for el in self.db.articles.find({"split": split}, projection=projection)]

        self.counter = 0

    def _process_article(self, article):
        """Process a single article and all its images."""
        article_data = []
        sections = article["parsed_section"]

        for pos in article["image_positions"]:
            if pos >= len(sections) or sections[pos]["type"] != "caption":
                continue

            caption = sections[pos]["text"].strip()
            if not caption:
                continue

            image_hash = sections[pos]["hash"]
            image_path = os.path.join(self.image_dir, f"{image_hash}.jpg")
            try:
                image = Image.open(image_path).convert("RGB")
            except (FileNotFoundError, OSError):
                logger.error(f"Could not open image at {image_path}")
                continue

            # Caption
            tokenized_caption = self.tokenizer(
                caption,
                max_length=self.max_length,
                truncation=True,
                # padding="max_length", # I think is doesn't really help and really sucks for performance
                return_tensors="pt",
            )
            caption_input_ids = tokenized_caption["input_ids"].to(self.device)
            caption_attention_mask = tokenized_caption["attention_mask"].to(self.device)
            outputs = self.roberta(caption_input_ids, attention_mask=caption_attention_mask)
            caption_embedding = outputs.last_hidden_state.squeeze()
            # Text
            text = self._get_context(article, pos)
            tokenized_text = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                # padding="max_length",
                return_tensors="pt",
            )
            text_input_ids = tokenized_text["input_ids"].to(self.device)
            text_attention_mask = tokenized_text["attention_mask"].to(self.device)
            outputs = self.roberta(text_input_ids, attention_mask=text_attention_mask)
            text_embedding = outputs.last_hidden_state.squeeze()

            # Image
            with torch.no_grad():
                image_tensor = self.image_transforms(image).unsqueeze(0).to(self.device)
                image_embedding = self.image_encoder(image_tensor)
                # shape: (1, 2048, 7, 7) -> (49, 1, 2048)
                image_embedding = image_embedding.squeeze(0)
                image_embedding = image_embedding.permute(1, 2, 0)
                image_embedding = image_embedding.reshape(-1, 1, 2048)
                # downsample 2048 -> 1024
                # shape: (49, 1, 1024)
                image_embedding = image_embedding[:, :, ::2].squeeze()

            # Faces
            if "facenet_details" in sections[pos]:
                tmp = torch.tensor(sections[pos]["facenet_details"]["embeddings"])
                faces = torch.zeros(tmp.shape[0], self.d_model)
                # upsample 512 -> 1024
                faces[:, :512] = tmp
            else:
                faces = torch.zeros(0, self.d_model)

            # Objects
            objects = self.db.objects.find_one({"_id": image_hash})
            if objects is not None and len(objects["object_features"]) != 0:
                objects = torch.tensor(objects["object_features"])
                # downsample 2048 -> 1024
                objects = objects[:, ::2]
            else:
                objects = torch.zeros(0, self.d_model)

            article_data.append(
                {
                    "image": image_embedding,
                    "caption": caption_embedding,
                    "context": text_embedding,
                    "faces": faces,
                    "objects": objects,
                }
            )

        return article_data

    def _get_context(self, article, pos):
        """Extract context from the article."""
        context = []
        if "main" in article["headline"]:
            context.append(article["headline"]["main"].strip())

        sections = article["parsed_section"]
        for i, section in enumerate(sections):
            if section["type"] == "paragraph":
                if pos - self.context_before <= i <= pos + self.context_after:
                    context.append(section["text"])

        return " ".join(context)

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, idx):
        if idx == 0:
            self.counter = 0

        image_article_queue = []

        # refill image_queue if it's empty
        while len(image_article_queue) == 0:
            self.counter += 1
            projection = ["parsed_section", "image_positions", "headline"]

            article = self.db.articles.find_one({"_id": self.article_ids[self.counter]}, projection=projection)
            image_article_queue.extend(self._process_article(article))

        image_article = image_article_queue.pop(0)
        return image_article


import time

start_time = time.time()

dataset = TaTDatasetReader("../data/nytimes/images_processed/", "localhost", split="valid")

print(len(dataset))

dataloader = DataLoader(dataset, batch_size=1)

counter = 0
for sample in dataloader:
    counter += 1
print(counter)

calculation_time = time.time() - start_time
print(f"Time in seconds: {calculation_time:.6f}")
