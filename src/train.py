from typing import Dict, Iterable

import torch
import yaml
from torch.utils.data import DataLoader
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
    )


def create_data_collator(tokenizer):
    """Create a data collator for batching."""

    def collate_fn(batch):
        if batch is Iterable:
            return batch[0]
        return batch

    return collate_fn


class BadNewsTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            tgt=inputs[:][0]["caption"],
            contexts=inputs[:][0]["contexts"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    config = load_config("src/config.yml")
    wandb.init(project=config["wandb"]["project"], config=config)

    tokenizer = RobertaTokenizer.from_pretrained(config["encoder"]["text_encoder"])
    print("Create train dataset")
    # init_dataset = TaTDatasetReader(
    #     image_dir=config["dataset"]["image_dir"],
    #     mongo_host=config["dataset"]["mongo_host"],
    #     mongo_port=config["dataset"]["mongo_port"],
    #     roberta_model=config["encoder"]["text_encoder"],
    #     max_length=config["training"]["max_target_positions"],
    #     device=config["training"]["device"],
    #     split="train",
    # )

    print("Create validation dataset")
    eval_dataset = TaTDatasetReader(
        image_dir=config["dataset"]["image_dir"],
        mongo_host=config["dataset"]["mongo_host"],
        mongo_port=config["dataset"]["mongo_port"],
        roberta_model=config["encoder"]["text_encoder"],
        max_length=config["training"]["max_target_positions"],
        device=config["training"]["device"],
        split="valid",
    )

    print("Create Model")
    model = create_model(config, tokenizer.vocab_size)
    model.to(config["training"]["device"])

    print("Load training arguments")
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        weight_decay=config["training"]["weight_decay"],
        logging_dir=config["training"]["logging_dir"],
        report_to="wandb",
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        warmup_steps=500,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=100,
    )

    print("Load data collator")
    data_collator = create_data_collator(tokenizer)

    trainer = BadNewsTrainer(
        model=model,
        args=training_args,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Start training")
    trainer.train()
    print("Start evaluation")
    trainer.evaluate()

    wandb.finish()


if __name__ == "__main__":
    main()
