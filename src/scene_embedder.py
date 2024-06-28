import torch
import torchvision.transforms as transforms
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from PIL import Image
from pymongo import MongoClient
from PIL import Image
import os

# DO NOT TOUCH
from resnet152Places365 import resnet152Places365


class ActivationExtractor:
    def __init__(self, model: torch.nn.Sequential, layer_from_end):
        self.model = model
        model.eval()
        self.layer_from_end = layer_from_end
        self.hook_handle = None
        self.activation = None

        self.layer_index = len(list(model.children())) - layer_from_end
        if self.layer_index < 0:
            raise ValueError(f"The model does not have {layer_from_end} layers")

        self.target_layer = list(model.children())[self.layer_index]
        self.hook_handle = self.target_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activation = output

    def get_activation(self, input):
        with torch.no_grad():
            _ = self.model(input)
        return self.activation


def connect():
    client = MongoClient("mongodb://root:secure_pw@localhost:27017/")
    return client


def imgage_path_generator(
    image_folder="./../input_data/uncompressed/images_processed/",
    image_extension=".jpg",
    sample=0,
):
    # print("get_images...")
    client = connect()

    # print("Getting correct collection...")
    db = client["nytimes_sample"]
    image_table = db["images"]

    documents = image_table.find().limit(sample)
    for doc in documents:
        yield (doc["_id"], os.path.join(image_folder, doc["_id"] + image_extension))


def embedding_saver_factory():
    db_name = "place_embeddings"
    client = connect()
    db = client[db_name]
    embedding_table = db["embeddings"]

    def store_embedding(key, embedding: torch.Tensor):
        embedding = embedding.numpy().tolist()
        record = {key: embedding}
        embedding_table.insert_one(record)

    return store_embedding


def load_model(device="cuda"):
    model = resnet152Places365.resnet152_places365
    model.load_state_dict(torch.load("./src/resnet152Places365/resnet152Places365.pth"))
    model.eval()
    model.to(device)
    return model


def preprocess_image(image: Image):
    # transform for image
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply the transformation
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    return image


def get_embedding(images: Image, layer_from_end=2, device="cuda") -> torch.tensor:
    model = load_model()
    activation_extractor = ActivationExtractor(model, layer_from_end=layer_from_end)

    # Preprocess the image
    image_tensor = preprocess_image(images).to(device)

    embeddings = activation_extractor.get_activation(image_tensor).to(device)

    return embeddings


if __name__ == "__main__":
    img_path_gen = imgage_path_generator(image_folder="./data/nytimes/sample/sample_images/")
    save_embedding = embedding_saver_factory()
    count = 0
    for img_hash, img_path in img_path_gen:
        image = Image.open(img_path)
        embedding = get_embedding(image, 2)
        embedding = embedding.cpu()
        save_embedding(img_hash, embedding)
        count += 1
        if count >= 4:
            break
