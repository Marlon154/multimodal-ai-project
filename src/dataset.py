import gc
import logging
import os
from typing import List

import torch
from PIL import Image
from pymongo import MongoClient
from torch._C import wait
from torch.nn.utils.rnn import pad_sequence
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
        mongo_password: str = "secure_pw",
        roberta_model: str = "roberta-large",
        max_length: int = 512,
        context_before: int = 8,
        context_after: int = 8,
        seed: int = 464896,
        split: str = "train",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        contexts: List[str] = ["article", "image", "faces", "objects"],
    ):
        self.client = MongoClient(f"mongodb://root:{mongo_password}@{mongo_host}:{mongo_port}/")
        self.db = self.client.nytimes
        self.context_before = context_before
        self.context_after = context_after

        self.preprocess = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        self.data = []
        self.d_model = 1024

        self.contexts = contexts

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
        projection = ["_id", "image_positions", "parsed_section"]
        articles = self.db.articles.find({"split": split}, projection=projection).limit(2)  # todo remove
        for article in tqdm(articles, desc="Article Preprocessing", total=434314):
            for pos in article["image_positions"]:
                caption = article["parsed_section"][pos]["text"].strip()
                if not caption:
                    continue

                # Image
                image_hash = article["parsed_section"][pos]["hash"]
                image_path = os.path.join(self.image_dir, f"{image_hash}.jpg")
                if not os.path.exists(image_path):
                    logger.error(f"Could not open image at {image_path}")
                    continue
                self.article_ids_image_pos.append(f"{article["_id"]}_{pos}")

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
        # if not caption:
        #     return get_empty_result()

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

        res = {
            "caption_tokenids": caption_input_ids,
            "caption_mask": caption_attention_mask,
            "caption": caption_embedding.cpu(),
            "contexts": [],
        }

        # Text
        if "article" in self.contexts:
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
            res["contexts"].append(text_embedding)

        # Image
        image_hash = sections[image_position]["hash"]
        if "image" in self.contexts:
            image_path = os.path.join(self.image_dir, f"{image_hash}.jpg")
            image = Image.open(image_path).convert("RGB")

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
            res["contexts"].append(image_embedding)

        # Faces
        if "faces" in self.contexts:
            if "facenet_details" in sections[image_position]:
                tmp = torch.tensor(sections[image_position]["facenet_details"]["embeddings"])
                faces = torch.zeros(tmp.shape[0], self.d_model)
                # upsample 512 -> 1024
                faces[:, :512] = tmp
            else:
                faces = torch.zeros(0, self.d_model)
            res["contexts"].append(faces)

        # Objects
        if "objects" in self.contexts:
            objects = self.db.objects.find_one({"_id": image_hash})
            if objects is not None and len(objects["object_features"]) != 0:
                objects = torch.tensor(objects["object_features"])
                # downsample 2048 -> 1024
                objects = objects[:, ::2]  # todo
            else:
                objects = torch.zeros(0, self.d_model)
            res["contexts"].append(objects)

        # scene embeddings
        if "scene_embeddings" in self.contexts:
            scene_embeddings = self.db.place_embeddings.find_one({"_id": image_hash})
            if scene_embeddings is not None:
                # adjust to scene embeddings
                pass
                # objects = torch.tensor(objects["object_features"])
                # # downsample 2048 -> 1024
                # objects = objects[:, ::2]  # todo
            else:
                scene_embeddings = torch.zeros(0, self.d_model)
            res["contexts"].append(scene_embeddings)

        return res

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


def collate_fn(batch):
    caption_tokenids = [item["caption_tokenids"].squeeze(0) for item in batch]
    captions = [item["caption"] for item in batch]
    contexts = [item["contexts"] for item in batch]

    # Pad caption token ids
    caption_tokenids_padded = pad_sequence(
        caption_tokenids, batch_first=True, padding_value=1
    )  # 1 is the pad token ID for RoBERTa
    caption_mask = (caption_tokenids_padded != 1).long()

    # Pad captions and contexts
    max_caption_len = max(cap.size(0) for cap in captions)
    max_context_lens = [max(ctx[i].size(0) for ctx in contexts) for i in range(4)]

    captions_padded = torch.stack(
        [torch.cat([cap, torch.zeros(max_caption_len - cap.size(0), cap.size(1))]) for cap in captions]
    )
    contexts_padded = [
        torch.stack(
            [torch.cat([ctx[i], torch.zeros(max_context_lens[i] - ctx[i].size(0), ctx[i].size(1))]) for ctx in contexts]
        )
        for i in range(4)
    ]

    return {
        "caption_tokenids": caption_tokenids_padded,
        "caption_mask": caption_mask,
        "caption": captions_padded,
        "contexts": contexts_padded,
    }
