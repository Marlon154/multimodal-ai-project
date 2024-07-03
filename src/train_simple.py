import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
import yaml
from typing import Dict
from dataset import TaTDatasetReader, collate_fn
from model import BadNews
from transformers import RobertaTokenizer


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


def setup(rank, world_size):
    if rank == 0:
        wandb.init(project="MAI-Project", mode="offline")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, config):
    setup(rank, world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    eval_dataset = TaTDatasetReader(
        image_dir=config["dataset"]["image_dir"],
        mongo_host=config["dataset"]["mongo_host"],
        mongo_port=config["dataset"]["mongo_port"],
        roberta_model=config["encoder"]["text_encoder"],
        max_length=config["training"]["max_target_positions"],
        device=device,
        split="train",
    )

    tokenizer = RobertaTokenizer.from_pretrained(config["encoder"]["text_encoder"])

    model = create_model(config, tokenizer.vocab_size)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    batch_size = config["training"]["train_batch_size"] // world_size
    sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    num_train_epochs = config["training"]["num_train_epochs"]

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), config["training"]["learning_rate"])

    if rank == 0:
        print("Start training")
    print(len(eval_dataset))
    for epoch in range(num_train_epochs):
        model.train()
        sampler.set_epoch(epoch)
        train_loss = []

        for sample in tqdm(dataloader, disable=rank != 0):
            caption = sample["caption"].to(device)
            contexts = [context.to(device) for context in sample["contexts"]]
            caption_tokenids = sample["caption_tokenids"].to(device)

            optimizer.zero_grad()

            output = model(tgt_embeddings=caption, contexts=contexts)

            loss = loss_fn(output.view(-1, output.size(-1)), caption_tokenids.view(-1))
            loss.backward()

            optimizer.step()

            loss = loss.item()
            train_loss.append(loss)

        if rank == 0:
            wandb.log({"epoch": epoch, "loss": sum(train_loss) / 16})

    if rank == 0:
        torch.save(model.module.state_dict(), "/home/ml-stud14/mai-data/output/model.pt")

    cleanup()


def main():
    config = load_config("src/config.yml")
    world_size = 4
    torch.multiprocessing.spawn(train, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    wandb.init(project="MAI-Project")
    main()