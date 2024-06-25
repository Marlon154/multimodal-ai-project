import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import Trainer, TrainingArguments
from datasets import Dataset

from tell.models import TransformerFacesObjectModel
from tell.data.dataset_readers.nytimes_faces_ner_matched import NYTimesFacesNERMatchedReader


def create_dataset(data_path, tokenizer, image_processor):
    reader = NYTimesFacesNERMatchedReader(
        tokenizer=tokenizer,
        token_indexers={'roberta': tokenizer},
        image_dir='data/nytimes/images_processed',
        use_caption_names=False,
        use_objects=True
    )

    instances = list(reader.read(data_path))

    def process_instance(instance):
        return {
            'context': instance.fields['context'].tokens,
            'image': image_processor(instance.fields['image'].image),
            'caption': instance.fields['caption'].tokens,
            'face_embeds': instance.fields['face_embeds'].array,
            'obj_embeds': instance.fields['obj_embeds'].array,
            'target': instance.fields['caption'].tokens[:, 1:]  # Shift right for teacher forcing
        }

    return Dataset.from_list([process_instance(instance) for instance in instances])


def main():
    from transformers import RobertaTokenizer
    from torchvision.transforms import Compose, Normalize, ToTensor

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


if __name__ == "__main__":
    main()