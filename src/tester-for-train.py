# import torch
# import torch.nn as nn
# from transformers import RobertaModel, RobertaTokenizer
#
# from decoder import TransformerDecoder, TransformerDecoderLayer
# from model import BadNews
#
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
# roberta_model = RobertaModel.from_pretrained("roberta-large")
#
# text = "A peacock roaming on the island of Lokrum, near the city of Dubrovnik in Croatia. "
#
# # tokenize and encode
# inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True, padding="max_length")
# input_ids = inputs["input_ids"][:4]
# print(f"Input: {input_ids}")
# attention_mask = inputs["attention_mask"]
# print(f"Attention Mask: {attention_mask}")
# print(f"Input Shape: {input_ids.shape}")
# # get contextualized embeddings
# with torch.no_grad():
#     current_seq = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long, device="cpu")
#     # current_seq[:, 1:] = tokenizer.pad_token_id
#     outputs = roberta_model(input_ids=current_seq)
#     tgt_embeddings = outputs.last_hidden_state.to("cpu")  # Transpose to match the expected shape
#     print(f"Embedding Shape: {tgt_embeddings.shape}")
#
# # Create the model
# d_model = 1024
# num_contexts = 4
# num_heads = 16
# num_decoder_layers = 4
# d_feedforward = 4096
# vocab_size = tokenizer.vocab_size
#
# context_lengths = [20, 5, 0, 2]
# context_dims = [1024, 512, 256, 512]
# contexts = [torch.rand((1, context_lengths[idx], d_model), device="cpu") for idx in range(num_contexts)]
# # print(f"Shape of first context: {contexts[0].shape}")
# # print(f"ncontexts: {num_contexts}")
#
# # decoder = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads)
# # out = decoder.forward(tgt_embeddings, tgt_embeddings)
# # print(f"Torch Model Output: {out.shape}")
# #
# # decoder_self = TransformerDecoderLayer(d_model=d_model, nhead=num_heads, ncontexts=num_contexts)
# # out_self = decoder_self.forward(tgt_embeddings, contexts)
# # print(f"Own Model Output: {out_self.shape}")
#
# # Forward Pass:
# # Positional Embedding -> Self-Attention -> Norm -> Cross-Attention (Self + Contexts) for multiple contexts -> Norm -> Feed Foward -> Norm -> Repeat n times -> Linear Layer
# model = BadNews(
#     vocab_size=vocab_size,
#     d_model=d_model,
#     nhead=num_heads,
#     num_decoder_layers=num_decoder_layers,
#     dim_feedforward=d_feedforward,
#     max_seq_length=512,
#     ncontexts=num_contexts,
#     device="cpu",
# )
# tmp = torch.load("/home/ml-stud14/mai-data/output/run4/checkpoint_epoch_0_batch_11000.pt", map_location="cpu")
# state_dict = tmp["model_state_dict"]
# state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# model.load_state_dict(state_dict)
# output = model.forward(tgt_embeddings, contexts)
# predicted_token_ids = torch.argmax(output, dim=-1).squeeze()
# # predicted_token_ids = torch.tensor(predicted_token_ids.tolist())
# prev_token_embedding = roberta_model(input_ids=[tokenizer.bos_token_id, tokenizer.bos_token_id])
#
# with torch.no_grad():
#     for _ in range(32):
#         output = model.forward(prev_token_embedding.last_hidden_state, contexts)
#         predicted_token = torch.argmax(output, dim=-1).squeeze()
#         print("output", output.shape)
#         print("predicted_token", predicted_token.shape)
#         print("predicted", tokenizer.decode(predicted_token))
#         prev_token_embedding = roberta_model(input_ids=torch.tensor(predicted_token))
#         # predicted_token_ids = torch.cat((predicted_token_ids, torch.argmax(output, dim=-1).squeeze().unsqueeze(0)))
#         # print(f"Predicted Token IDs: {predicted_token_ids}")
#
#         tgt_embeddings = torch.cat((tgt_embeddings, predicted_token[-1]), dim=0)
#
#
# shift_right = output[:, :-1, :]
# labels_shifted_right = input_ids[:, 1:]
# # print("labels_shifted_right", labels_shifted_right)
# # print("shifted_right argmax", torch.argmax(shift_right, dim=-1))
# # print("labels_shifted_right.shape", labels_shifted_right.shape)
# # print(f"Predicted Token IDs: {predicted_token_ids}")
#
# decoded_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
# print("        text:", text)
# print("decoded text:", decoded_text)
#
# loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#
# # print(f"Predicted Token IDs shape: {predicted_token_ids.shape}")
# # print(f"Model Output Shape: {shift_right.shape}")
# # print(f"Model Output: {shift_right}")
# # print(f"Input IDs Shape: {input_ids.shape}")
# # print(f"Input IDs Shape view: {input_ids.view(-1).shape}")
# loss = loss_fn(shift_right.view(-1, shift_right.size(-1)), labels_shifted_right.view(-1))
#
# print(f"Loss: {loss}")
#
#
# # output = model.forward(tgt_embeddings, contexts)
# # predicted_token_ids = torch.argmax(output, dim=-1).squeeze()
# # # predicted_token_ids = torch.tensor(predicted_token_ids.tolist())
# #
# # shift_right = output[:, :-1, :]
# # labels_shifted_right = input_ids[:, 1:]
# # print("labels_shifted_right", labels_shifted_right)
# # print("shifted_right argmax", torch.argmax(shift_right, dim=-1))
# # # print("labels_shifted_right.shape", labels_shifted_right.shape)
# # print(f"Predicted Token IDs: {predicted_token_ids}")
# #
# # decoded_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
# # loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
# #
# # # print(f"Predicted Token IDs shape: {predicted_token_ids.shape}")
# # print(f"Model Output Shape: {shift_right.shape}")
# # print(f"Model Output: {shift_right}")
# # # print(f"Input IDs Shape: {input_ids.shape}")
# # # print(f"Input IDs Shape view: {input_ids.view(-1).shape}")
# # loss = loss_fn(shift_right.view(-1, shift_right.size(-1)), labels_shifted_right.view(-1))
# #
# # print(f"Loss: {loss}")
# # print(decoded_text)

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

from dataset import TaTDatasetReader
from decoder import TransformerDecoder, TransformerDecoderLayer
from model import BadNews

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
roberta_model = RobertaModel.from_pretrained("roberta-large")

# Create the model
d_model = 1024
num_contexts = 4
num_heads = 16
num_decoder_layers = 4
d_feedforward = 4096
vocab_size = tokenizer.vocab_size

context_lengths = [20, 5, 0, 2]
context_dims = [1024, 512, 256, 512]
contexts = [torch.rand((1, context_lengths[idx], d_model), device="cpu") for idx in range(num_contexts)]

model = BadNews(
    vocab_size=vocab_size,
    d_model=d_model,
    nhead=num_heads,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=d_feedforward,
    max_seq_length=512,
    ncontexts=num_contexts,
    device="cpu",
)

tmp = torch.load("/home/ml-stud14/mai-data/output/run4/checkpoint_epoch_0_batch_11500.pt", map_location="cpu")
state_dict = tmp["model_state_dict"]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)


# text = "The cat sat on the mat"
# inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
# init_token_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]
# print(f"Input: {init_token_ids}")
init_token_ids = torch.full((1, 1), tokenizer.pad_token_id, device="cpu")
init_token_ids[:, 0] = tokenizer.bos_token_id
attention_mask = torch.zeros((1, 1))
attention_mask[:, 0] = 1

# get contextualized embeddings

for i in range(32):

    with torch.no_grad():
        outputs = roberta_model(init_token_ids, attention_mask=attention_mask)

    tgt_embeddings = outputs.last_hidden_state.to("cpu")  # Transpose to match the expected shape
    # print(f"Embedding Shape: {tgt_embeddings.shape}")
    output = model.forward(tgt_embeddings, contexts)
    predicted_token_id = torch.argmax(output, dim=-1)
    # print(f"Predicted Token IDs: {predicted_token_id}")
    # print(f"Predicted Token IDs shape: {predicted_token_id.shape}")
    decoded_text = tokenizer.decode(predicted_token_id.squeeze(), skip_special_tokens=True)
    print("Predicted token", decoded_text)
    print(f"Predicted Token IDs: {predicted_token_id}")
    init_token_ids = torch.cat((init_token_ids, predicted_token_id), dim=1)
    print(f"Predicted Token IDs: {init_token_ids}")
    tmp = torch.full((1, i + 1), tokenizer.pad_token_id, device="cpu")
    init_token_ids[:, 0] = tokenizer.bos_token_id
    attention_mask = torch.ones((1, i + 1))
    attention_mask[:, 0] = 1
# init_token_ids = torch.full((1, 2), tokenizer.pad_token_id, device="cpu")
# init_token_ids[:, 1] = predicted_token_ids
# attention_mask = torch.ones((1, 2))
# with torch.no_grad():
#     outputs = roberta_model(init_token_ids, attention_mask=attention_mask)
#
# tgt_embeddings = outputs.last_hidden_state.to("cpu")  # Transpose to match the expected shape
# print(f"Embedding Shape: {tgt_embeddings.shape}")
#
# output = model.forward(tgt_embeddings, contexts)
# predicted_token_ids = torch.argmax(output, dim=-1).squeeze()
#
# print(f"Predicted Token IDs: {predicted_token_ids}")
# print(f"Predicted Token IDs shape: {predicted_token_ids.shape}")
# decoded_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
# print(decoded_text)
