import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from datasets import load_dataset
from torchvision.models import resnet152
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLOv10
from mtcnn import MTCNN


class TransformerDecoderModel:
    def __init__(self, model_name="roberta-base", dataset_name="nytimes_faces_ner_matched", output_dir="./expt/nytimes"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self.dataset = load_dataset(self.dataset_name)

        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=0.0001,
            weight_decay=0.00001,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=0.000001,
            max_grad_norm=0.1,
            num_train_epochs=100,
            lr_scheduler_type="linear",
            warmup_ratio=0.05,
            logging_steps=512,
            evaluation_strategy="steps",
            eval_steps=4376,
            save_strategy="steps",
            save_steps=4376,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            no_cuda=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["valid"],
            tokenizer=self.tokenizer,
        )

    def train(self):
        self.trainer.train()

    def evaluate(self):
        self.trainer.evaluate()


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, context):
        attn_output, _ = self.mha(x, context, context)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])

    def forward(self, x, context):
        for layer in self.layers:
            x = layer(x, context)
        return x

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dff, article_length, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(article_length, d_model)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, rate)
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, context):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.decoder(x, context)
        x = self.final_layer(x)
        return x

    @staticmethod
    def positional_encoding(length, d_model):
        positions = torch.arange(length).unsqueeze(1)
        divisions = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding = torch.zeros(length, d_model)
        pos_encoding[:, 0::2] = torch.sin(positions * divisions)
        pos_encoding[:, 1::2] = torch.cos(positions * divisions)
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding


class TransformAndTellModel(nn.Module):
    def __init__(self, vocab_size, article_length=512, d_model=512, num_layers=6, num_heads=8, dff=2048, rate=0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.resnet = resnet152(pretrained=True)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        self.yolo = YOLOv10.from_pretrained("jameslahm/yolov10x")
        self.mtcnn = MTCNN()
        self.image_encoder = nn.Linear(2048, d_model)
        self.face_encoder = nn.Linear(512, d_model)
        self.object_encoder = nn.Linear(2048, d_model)
        self.caption_generator = CaptionGenerator(vocab_size, d_model, num_layers, num_heads, dff, article_length, rate)

    def forward(self, article, image):
        # Encode article using RoBERTa
        article_embeddings = self.roberta(article)[0]

        # Encode image using ResNet-152
        image_features = self.resnet(image)
        image_embeddings = self.image_encoder(image_features)

        # Detect faces using MTCNN and encode using FaceNet
        faces = self.mtcnn(image)
        if faces is not None:
            face_features = self.facenet(faces)
            face_embeddings = self.face_encoder(face_features)
        else:
            face_embeddings = torch.zeros((1, self.face_encoder.out_features))

        # Detect objects using YOLOv10 and encode using ResNet-152
        objects = self.yolo(image)
        if objects is not None:
            object_features = self.resnet(objects)
            object_embeddings = self.object_encoder(object_features)
        else:
            object_embeddings = torch.zeros((1, self.object_encoder.out_features))

        # Concatenate all embeddings
        context = torch.cat((article_embeddings, image_embeddings, face_embeddings, object_embeddings), dim=1)

        # Generate caption using transformer decoder
        caption = self.caption_generator(article, context)

        return caption


# Load the dataset
dataset = load_dataset("nytimes_faces_ner_matched",
                       tokenizer=RobertaTokenizer.from_pretrained("roberta-base"),
                       image_dir="data/nytimes/images_processed",
                       use_caption_names=False,
                       use_objects=True)

# Define the model
model = TransformAndTellModel(vocab_size=50265, article_length=512)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./expt/nytimes",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=0.0001,
    weight_decay=0.00001,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=0.000001,
    max_grad_norm=0.1,
    num_train_epochs=100,
    lr_scheduler_type="linear",
    warmup_ratio=0.05,
    logging_steps=512,
    evaluation_strategy="steps",
    eval_steps=4376,
    save_strategy="steps",
    save_steps=4376,
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    no_cuda=False,
)
