import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import Trainer, TrainingArguments, RobertaTokenizer
from datasets import Dataset
import wandb
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from tell.models import TransformerFacesObjectModel
from tell.data.dataset_readers.nytimes_faces_ner_matched import NYTimesFacesNERMatchedReader


def create_dataset(split, tokenizer, image_processor):
    # Initialize the NYTimesFacesNERMatchedReader
    mongo_host = os.environ.get('MONGO_HOST', 'localhost')
    mongo_port = int(os.environ.get('MONGO_PORT', 27017))
    image_path = str(os.environ.get('IMAGE_PATH', "./databases/nytimes"))
    reader = NYTimesFacesNERMatchedReader(
        tokenizer=tokenizer,
        token_indexers={"tokens": tokenizer},
        image_dir=image_path,
        mongo_host=mongo_host,
        mongo_port=mongo_port,
        use_caption_names=True,
        use_objects=True,
        n_faces=None,
        lazy=True
    )

    # Create a list to store the processed examples
    examples = []

    # Iterate through the dataset
    for instance in tqdm(reader.read(split), desc=f"Processing {split} dataset"):
        # Extract relevant information from the instance
        context = instance.fields["context"].tokens
        names = [name.tokens for name in instance.fields["names"].field_list]
        image = instance.fields["image"].image
        caption = instance.fields["caption"].tokens
        face_embeds = instance.fields["face_embeds"].array
        obj_embeds = instance.fields["obj_embeds"].array if "obj_embeds" in instance.fields else None
        metadata = instance.fields["metadata"].metadata

        # Process the image
        processed_image = image_processor(image)

        # Create a dictionary for the example
        example = {
            "context": tokenizer.convert_tokens_to_ids(context),
            "names": [tokenizer.convert_tokens_to_ids(name) for name in names],
            "image": processed_image,
            "caption": tokenizer.convert_tokens_to_ids(caption),
            "face_embeds": face_embeds,
            "obj_embeds": obj_embeds,
            "metadata": metadata
        }

        examples.append(example)

    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict({
        "context": [ex["context"] for ex in examples],
        "names": [ex["names"] for ex in examples],
        "image": [ex["image"] for ex in examples],
        "caption": [ex["caption"] for ex in examples],
        "face_embeds": [ex["face_embeds"] for ex in examples],
        "obj_embeds": [ex["obj_embeds"] for ex in examples],
        "metadata": [ex["metadata"] for ex in examples]
    })

    return dataset

def main():
    # Initialize wandb
    wandb.init(project="MAI-Project")

    # Load tokenizer and image processor
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    image_processor = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = create_dataset('train', tokenizer, image_processor)
    val_dataset = create_dataset('valid', tokenizer, image_processor)

    # Define model configuration
    model_config = {
        'model_name': 'roberta-base',
        'decoder_layers': 4,
        'hidden_size': 1024,
        'attention_dim': 1024,
        'decoder_ffn_embed_dim': 4096,
        'decoder_attention_heads': 16,
        'dropout': 0.1,
        'weigh_bert': True,
        'bert_layers': 13,
        'padding_value': 1,
        'vocab_size': 50265,
    }

    # Log model config to wandb
    wandb.config.update(model_config)

    # Initialize model
    model = TransformerFacesObjectModel(model_config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=2188,
        weight_decay=0.00001,
        logging_dir='./logs',
        logging_steps=512,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True,
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

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()