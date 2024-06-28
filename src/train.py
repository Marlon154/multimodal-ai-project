import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import Trainer, TrainingArguments, RobertaTokenizer, RobertaModel
from datasets import Dataset
import wandb
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import torch

from tell.models import TransformerFacesObjectModel, DynamicConvFacesObjectsDecoder
from tell.data.dataset_readers.nytimes_faces_ner_matched import NYTimesFacesNERMatchedReader
from tell.modules import AdaptiveSoftmax
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder


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
        'max_target_positions': 512,
        'share_decoder_input_output_embed': True,
        'decoder_conv_dim': 1024,
        'decoder_glu': True,
        'decoder_conv_type': 'dynamic',
        'weight_softmax': True,
        'weight_dropout': 0.1,
        'relu_dropout': 0.1,
        'input_dropout': 0.1,
        'decoder_normalize_before': True,
        'attention_dropout': 0.1,
        'decoder_kernel_size_list': [3, 7, 15, 31],
        'adaptive_softmax_cutoff': None,
        'tie_adaptive_weights': False,
        'adaptive_softmax_dropout': 0,
        'tie_adaptive_proj': False,
        'adaptive_softmax_factor': 0,
        'section_attn': False,
        'swap': False
    }

    # Log model config to wandb
    wandb.config.update(model_config)

    # Create vocabulary
    vocab = Vocabulary()

    # Create embedder using Hugging Face transformers
    roberta_model = RobertaModel.from_pretrained('roberta-base')

    # Wrap the RoBERTa model in a simple PyTorch module to make it compatible
    class RobertaEmbedder(torch.nn.Module):
        def __init__(self, roberta_model):
            super().__init__()
            self.roberta = roberta_model

        def forward(self, tokens):
            return self.roberta(tokens)[0]  # return the last hidden states

        def get_output_dim(self):
            return self.roberta.config.hidden_size

    embedder = RobertaEmbedder(roberta_model)

    # Create decoder (modify if necessary to accept the new embedder type)
    decoder = DynamicConvFacesObjectsDecoder(
        vocab=vocab,
        embedder=embedder,
        max_target_positions=model_config['max_target_positions'],
        dropout=model_config['dropout'],
        share_decoder_input_output_embed=model_config['share_decoder_input_output_embed'],
        decoder_output_dim=model_config['hidden_size'],
        decoder_conv_dim=model_config['decoder_conv_dim'],
        decoder_glu=model_config['decoder_glu'],
        decoder_conv_type=model_config['decoder_conv_type'],
        weight_softmax=model_config['weight_softmax'],
        decoder_attention_heads=model_config['decoder_attention_heads'],
        weight_dropout=model_config['weight_dropout'],
        relu_dropout=model_config['relu_dropout'],
        input_dropout=model_config['input_dropout'],
        decoder_normalize_before=model_config['decoder_normalize_before'],
        attention_dropout=model_config['attention_dropout'],
        decoder_ffn_embed_dim=model_config['decoder_ffn_embed_dim'],
        decoder_kernel_size_list=model_config['decoder_kernel_size_list'],
        adaptive_softmax_cutoff=model_config['adaptive_softmax_cutoff'],
        tie_adaptive_weights=model_config['tie_adaptive_weights'],
        adaptive_softmax_dropout=model_config['adaptive_softmax_dropout'],
        tie_adaptive_proj=model_config['tie_adaptive_proj'],
        adaptive_softmax_factor=model_config['adaptive_softmax_factor'],
        decoder_layers=model_config['decoder_layers'],
        final_norm=True,
        padding_idx=model_config['padding_value'],
        namespace='target_tokens',
        vocab_size=model_config['vocab_size'],
        section_attn=model_config['section_attn'],
        swap=model_config['swap']
    )

    # Create criterion
    criterion = AdaptiveSoftmax(
        vocab_size=model_config['vocab_size'],
        cutoff=model_config['adaptive_softmax_cutoff'] or [model_config['vocab_size']],
        dropout=model_config['adaptive_softmax_dropout'],
        factor=model_config['adaptive_softmax_factor']
    )

    # Initialize model
    model = TransformerFacesObjectModel(
        vocab=vocab,
        decoder=decoder,
        criterion=criterion,
        **model_config
    )

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
        train_dataset=val_dataset,  # Using val_dataset for both train and eval for this example
        eval_dataset=val_dataset,
    )

    # Start training
    trainer.train()

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
