import yaml
import wandb
from transformers import TrainingArguments, Trainer, RobertaModel
from model import BadNews
from torch.utils.data import DataLoader
import torch
from typing import Dict
from dataset import TaTDatasetReader
from transformers import RobertaTokenizer


def load_config(config_path: str) -> Dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error loading config: {exc}")


def create_model(config: Dict, vocab_size) -> BadNews:
    """Create a BadNews model instance from configuration."""
    return BadNews(
        vocab_size=vocab_size,
        d_model=config['decoder']['hidden_size'],
        nhead=config['decoder']['attention_heads'],
        num_decoder_layers=config['decoder']['layers'],
        dim_feedforward=config['decoder']['ffn_embed_dim'],
        max_seq_length=config['training']['max_target_positions'],
        ncontexts=config['decoder']['layers'],
    )


def create_data_collator(tokenizer):
    """Create a data collator for batching."""

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        context_ids = torch.stack([item['context_ids'] for item in batch])
        context_mask = torch.stack([item['context_mask'] for item in batch])
        context_embeddings = torch.stack([item['context_embeddings'] for item in batch])
        caption_ids = torch.stack([item['caption_ids'] for item in batch])
        caption_mask = torch.stack([item['caption_mask'] for item in batch])

        return {
            'images': images,
            'context_ids': context_ids,
            'context_mask': context_mask,
            'context_embeddings': context_embeddings,
            'caption_ids': caption_ids,
            'caption_mask': caption_mask,
        }

    return collate_fn


class BadNewsTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            images=inputs['images'],
            context_embeddings=inputs['context_embeddings'],
            context_mask=inputs['context_mask'],
            caption_ids=inputs['caption_ids'],
            caption_mask=inputs['caption_mask']
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def main():
    config = load_config("src/config.yml")
    wandb.init(project=config['wandb']['project'], config=config)

    tokenizer = RobertaTokenizer.from_pretrained(config['encoder']['text_encoder'])
    roberta_model = RobertaModel.from_pretrained(config['encoder']['text_encoder'])
    # todo add image encoder
    image_encoder = "resnet152 todo"
    init_dataset = TaTDatasetReader(
        image_encoder=image_encoder,
        image_dir=config['dataset']['image_dir'],
        mongo_host=config['dataset']['mongo_host'],
        mongo_port=config['dataset']['mongo_port'],
        roberta_model=config['encoder']['text_encoder'],
        max_length=config['training']['max_target_positions'],
        device=config['training']['device'],
        split='train',
    )

    eval_dataset = TaTDatasetReader(
        image_dir=config['dataset']['image_dir'],
        mongo_host=config['dataset']['mongo_host'],
        mongo_port=config['dataset']['mongo_port'],
        roberta_model=config['encoder']['text_encoder'],
        max_length=config['training']['max_target_positions'],
        device=config['training']['device'],
        split='valid',
    )

    model = create_model(config, tokenizer.vocab_size)
    model.to(config['training']['device'])

    eval_dataset.load_data('valid')  # Load validation data

    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        weight_decay=config['training']['weight_decay'],
        logging_dir=config['training']['logging_dir'],
        report_to='wandb',
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        warmup_steps=500,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=100,
    )

    data_collator = create_data_collator(tokenizer)

    trainer = BadNewsTrainer(
        model=model,
        args=training_args,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.evaluate()

    wandb.finish()


if __name__ == "__main__":
    main()
