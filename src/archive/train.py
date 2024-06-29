import argparse
import sys
import os
import yaml
from transformers import Trainer, TrainingArguments, RobertaTokenizer, RobertaModel
from datasets import Dataset, Features, Image
import wandb
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import torch
from dataset import TaTDatasetReader


def create_dataset(split, dataset_config, tokenizer, text_encoder):
    # Initialize the NYTimesFacesNERMatchedReader
    reader = TaTDatasetReader(
        image_dir=dataset_config['image_dir'],
        mongo_host=dataset_config['mongo_host'],
        mongo_port=dataset_config['mongo_port'],
        use_caption_names=dataset_config['use_caption_names'],
        use_objects=dataset_config['use_objects'],
        n_faces=dataset_config['n_faces'],
        max_length=dataset_config['max_length'],
        roberta_model=dataset_config['roberta_model'],
        device=dataset_config['training']['device']
    )
    # Create a list to store the processed examples
    examples = []

    # Iterate through the dataset
    for instance in tqdm(reader.read(split), desc=f"Processing {split} dataset"):
        context = instance.fields["context"].tokens
        names = [name.tokens for name in instance.fields["names"].field_list]
        image = instance.fields["image"].image
        caption = instance.fields["caption"].tokens
        face_embeds = instance.fields["face_embeds"].array
        obj_embeds = instance.fields["obj_embeds"].array if "obj_embeds" in instance.fields else None
        metadata = instance.fields["metadata"].metadata

        # Create a dictionary for the example
        example = {
            "context": tokenizer.convert_tokens_to_ids(context),
            "names": [tokenizer.convert_tokens_to_ids(name) for name in names],
            "image": image,
            "caption": tokenizer.convert_tokens_to_ids(caption),
            "face_embeds": face_embeds,
            "obj_embeds": obj_embeds,
            "metadata": metadata
        }

        examples.append(example)

    features = Features({
        "context": list,
        "names": list,
        "image": Image(),
        "face_embeds": list,
        "obj_embeds": list,
        "metadata": dict
    })
    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict(examples, features=features)

    # Set the format of the dataset to "torch"
    dataset = dataset.with_format("torch")
    return dataset


def main(config):
    wandb.init(config['wandb']['project'])
    wandb.config.update(config)

    # Load tokenizer and image processor
    tokenizer = RobertaTokenizer.from_pretrained(config["encoder"]["text_encoder"])
    embedder = RobertaModel.from_pretrained(config["encoder"]["text_encoder"])

    # Create datasets
    train_dataset = create_dataset('valid', config['dataset'], tokenizer, embedder)  # todo change to 'train' for real training
    val_dataset = create_dataset('valid', config['dataset'], tokenizer, embedder)

    # Create criterion
    # criterion = AdaptiveSoftmax(
    #     vocab_size=config['vocab_size'],
    #     cutoff=config['adaptive_softmax_cutoff'] or [config['vocab_size']],
    #     dropout=config['adaptive_softmax_dropout'],
    #     factor=config['adaptive_softmax_factor']
    # )

    # Initialize model
    model = ""

    trainer_config = config['trainer']
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=trainer_config['num_train_epochs'],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=2188,
        weight_decay=trainer_config['weight_decay'],
        logging_dir='./logs',
        logging_steps=512,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        greater_is_better=False,
        report_to="wandb",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Start training
    trainer.train()

    trainer.save_model()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('--config_path', type=str, default='/app/src/config.yml', help='The path to the configuration file')
    parser.add_argument('--device', type=str, default='cpu', help='The device to use for training')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    main(config)
