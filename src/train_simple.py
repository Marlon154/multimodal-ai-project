from typing import Dict, Iterable

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (RobertaModel, RobertaTokenizer, Trainer,
                          TrainingArguments)

import wandb
from dataset import TaTDatasetReader
from model import BadNews


def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error loading config: {exc}")


def create_model(config: Dict, vocab_size) -> BadNews:
    """Create a BadNews model instance from configuration."""
    return BadNews(
        vocab_size=vocab_size,
        d_model=config["decoder"]["hidden_size"],
        nhead=config["decoder"]["attention_heads"],
        num_decoder_layers=config["decoder"]["layers"],
        dim_feedforward=config["decoder"]["ffn_embed_dim"],
        max_seq_length=config["training"]["max_target_positions"],
        ncontexts=config["decoder"]["layers"],
        device=config["training"]["device"],
    )


def main():
    config = load_config("src/config.yml")

    eval_dataset = TaTDatasetReader(
        image_dir=config["dataset"]["image_dir"],
        mongo_host=config["dataset"]["mongo_host"],
        mongo_port=config["dataset"]["mongo_port"],
        roberta_model=config["encoder"]["text_encoder"],
        max_length=config["training"]["max_target_positions"],
        device=config["training"]["device"],
        split="train",
    )

    tokenizer = RobertaTokenizer.from_pretrained(config["encoder"]["text_encoder"])

    model = create_model(config, tokenizer.vocab_size)
    device = config["training"]["device"]
    model.to(device)

    dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=1)

    num_train_epochs = config["training"]["num_train_epochs"]
    learning_rate = 5e-5

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    print("Start training")
    for _ in range(num_train_epochs):
        train_loss = []

        for sample in tqdm(dataloader):
            caption, contexts = sample["caption"].to(device), sample["contexts"]
            caption_tokenids = sample["caption_tokenids"].to(device)
            for idx, context in enumerate(contexts):
                contexts[idx] = context.to(device)

            optimizer.zero_grad()

            output = model.forward(tgt_embeddings=caption, contexts=contexts)

            loss = loss_fn(output.squeeze(), caption_tokenids.squeeze())
            loss.backward()

            wandb.log({"loss": loss})

            optimizer.step()

            # only works for batch_size = 1 lol
            train_loss.append(loss)

    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    wandb.init(project="MAI-Project")
    main()
