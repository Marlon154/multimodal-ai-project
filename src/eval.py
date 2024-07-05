import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
import wandb
import yaml
from typing import Dict
from dataset import TaTDatasetReader, collate_fn
from model import BadNews
from transformers import RobertaTokenizer, RobertaModel
import os


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error loading config: {exc}")


def create_model(config: Dict, vocab_size) -> BadNews:
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


def load_model(config, model_path, from_checkpoint=False):
    """Load the model from a file."""
    tokenizer = RobertaTokenizer.from_pretrained(config["encoder"]["text_encoder"])
    roberta_model = RobertaModel.from_pretrained(config["encoder"]["text_encoder"], device_map=config["training"]["device"])
    model = create_model(config, tokenizer.vocab_size)  # Initialize your model class here
    if from_checkpoint:
        tmp = torch.load(model_path, map_location=config["training"]["device"])
        state_dict = tmp["model_state_dict"]
    else:
        state_dict = torch.load(model_path, map_location=config["training"]["device"])

    # Remove 'module.' prefix if it exists (happens when model was saved with DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(config["training"]["device"])
    return model, tokenizer, roberta_model


def evaluate(config):
    device = torch.device(f"cuda:0")

    eval_dataset = TaTDatasetReader(
        image_dir=config["dataset"]["image_dir"],
        mongo_host=config["dataset"]["mongo_host"],
        mongo_port=config["dataset"]["mongo_port"],
        roberta_model=config["encoder"]["text_encoder"],
        max_length=config["training"]["max_target_positions"],
        device=device,
        split="train",
    )
    dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=collate_fn)

    model, tokenizer, roberta_model = load_model(config, "/home/ml-stud14/mai-data/output/model.pt")

    print("Start testing")
    print(len(eval_dataset))
    model.eval()
    all_preds = []
    all_labels = []
    max_length = config["training"]["max_target_positions"]
    start_token_id = tokenizer.bos_token_id
    end_token_id = tokenizer.eos_token_id

    # with torch.no_grad():
    for i in range(1):
        for i, sample in enumerate(tqdm(dataloader, desc="Evaluating")):
            contexts = sample["contexts"]
            caption_tokenids = sample["caption_tokenids"].to(device)

            for idx, context in enumerate(contexts):
                contexts[idx] = context.to(device)

            # Initialize with start token
            current_seq = torch.full((1, 1), start_token_id, dtype=torch.long, device=device)

            generated_sequence = []

            for _ in range(max_length - 1):  # -1 because we started with one token
                # Get embeddings for the current sequence
                tgt_embeddings = roberta_model.embeddings.word_embeddings(current_seq)

                # Forward pass
                output = model.forward(tgt_embeddings=tgt_embeddings, contexts=contexts)

                # Get the next token prediction
                next_token_logits = output[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1)

                # Append the new token to the sequence
                current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                generated_sequence.append(next_token.item())

                # Check if we've hit the end token
                if next_token.item() == end_token_id:
                    break

            # Convert generated sequence to tensor
            generated_sequence = torch.tensor(generated_sequence, device=device)

            all_preds.extend(generated_sequence.cpu().numpy())
            all_labels.extend(caption_tokenids[0].cpu().numpy())

            # Log the examples
            if i < 10:
                pred_text = tokenizer.decode(generated_sequence)
                true_text = tokenizer.decode(caption_tokenids[0])
                print(f"Example {i}:")
                print(f"Prediction: {pred_text}")
                print(f"True caption: {true_text}")
                print()
                wandb.log({"example": i, "prediction": {pred_text}, "caption": true_text})

    # Pad sequences to the same length for metric calculation
    max_len = max(len(seq) for seq in all_preds + all_labels)
    all_preds_padded = [seq + [tokenizer.pad_token_id] * (max_len - len(seq)) for seq in all_preds]
    all_labels_padded = [seq.tolist() + [tokenizer.pad_token_id] * (max_len - len(seq)) for seq in all_labels]

    # Calculate metrics
    accuracy = accuracy_score(all_labels_padded, all_preds_padded)
    precision = precision_score(all_labels_padded, all_preds_padded, average='weighted')
    recall = recall_score(all_labels_padded, all_preds_padded, average='weighted')
    f1 = f1_score(all_labels_padded, all_preds_padded, average='weighted')

    # Decode
    all_true_texts = tokenizer.decode(all_preds)
    all_pred_texts = tokenizer.decode(all_labels)

    # Calculate BLEU-4
    smoothing_function = SmoothingFunction().method4
    bleu4 = corpus_bleu(all_true_texts, all_pred_texts, smoothing_function=smoothing_function)

    # Calculate ROUGE score
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(" ".join(sum(all_true_texts, [])), " ".join(sum(all_pred_texts, [])))
    
    rouge1 = rouge_scores['rouge1'].fmeasure
    rouge2 = rouge_scores['rouge2'].fmeasure
    rougeL = rouge_scores['rougeL'].fmeasure

    # Log metrics
    wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "bleu4": bleu4, "rouge": rouge_scores})
    wandb.finish()


if __name__ == "__main__":
    wandb.init(project="MAI-Project")
    config = load_config("src/config.yml")
    evaluate(config)
