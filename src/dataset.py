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

        self.article_ids_image_pos = []
        # load article ids
        projection = ["_id", "image_positions"]
        articles = self.db.articles.find({"split": split}, projection=projection).limit(100000)
        for article in tqdm(articles, desc="Article Preprocessing"):
            for pos in article["image_positions"]:
                # TODO: clean up articles with no captions or images which don't exist
                self.article_ids_image_pos.append(f"{article["_id"]}_{pos}")

        self.counter = 0

    def _process_image(self, article, image_position):
        """Process a single image of one article."""

        def get_empty_result():
            return {
                "caption_tokenids": torch.zeros(0, 50265),
                "caption": torch.zeros(0, 1024),
                "contexts": [
                    torch.zeros(0, 1024),
                    torch.zeros(0, 1024),
                    torch.zeros(0, 1024),
                    torch.zeros(0, 1024),
                ],
            }

        sections = article["parsed_section"]

        caption = sections[image_position]["text"].strip()
        if not caption:
            return get_empty_result()

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
        text = self._get_context(article, image_position)
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
        image_hash = sections[image_position]["hash"]
        image_path = os.path.join(self.image_dir, f"{image_hash}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            logger.error(f"Could not open image at {image_path}")
            return get_empty_result()

        with torch.no_grad():
            image_tensor = self.image_transforms(image).unsqueeze(0).to(self.device)
            image_embedding = self.image_encoder(image_tensor)
            # shape: (1, 2048, 7, 7) -> (49, 1, 2048)
            image_embedding = image_embedding.squeeze(0)
            image_embedding = image_embedding.permute(1, 2, 0)
            image_embedding = image_embedding.reshape(-1, 1, 2048)
            # downsample 2048 -> 1024
            # shape: (49, 1024)
            image_embedding = image_embedding[:, :, ::2].squeeze()

        # Faces
        if "facenet_details" in sections[image_position]:
            tmp = torch.tensor(sections[image_position]["facenet_details"]["embeddings"])
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

        return {
            "caption_tokenids": caption_input_ids,
            "caption": caption_embedding.cpu(),
            "contexts": [
                text_embedding.cpu(),
                image_embedding.cpu(),
                faces.cpu(),
                objects.cpu(),
            ],
        }

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
        return len(self.article_ids_image_pos)

    def __getitem__(self, idx):
        article_id, image_pos = self.article_ids_image_pos[idx].split("_")

        projection = ["parsed_section", "image_positions", "headline"]

        article = self.db.articles.find_one({"_id": article_id}, projection=projection)

        return self._process_image(article, int(image_pos))
