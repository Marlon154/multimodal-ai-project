import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from pymongo import MongoClient
import os
from torch.utils.data import Dataset, DataLoader

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
    client = MongoClient("mongodb://root:secure_pw@mongo_db:27017/")
    return client


class ImageDataset(Dataset):
    def __init__(self, image_folder="/data/images/", image_extension=".jpg", sample=0):
        self.image_folder = image_folder
        self.image_extension = image_extension
        self.client = connect()
        self.db = self.client["nytimes"]
        self.image_table = self.db["images"]
        self.documents = list(self.image_table.find().limit(sample))

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.documents[idx]
        img_path = os.path.join(self.image_folder, doc["_id"] + self.image_extension)
        image = Image.open(img_path).convert('RGB')
        image = preprocess_image(image)
        return doc["_id"], image


def embedding_saver_factory():
    db_name = "nytimes"
    client = connect()
    db = client[db_name]
    embedding_table = db["place_embeddings"]

    def store_embeddings(keys, embeddings: torch.Tensor):
        embeddings = embeddings.cpu().numpy().tolist()
        records = [{key: embedding} for key, embedding in zip(keys, embeddings)]
        embedding_table.insert_many(records)

    return store_embeddings


def load_model(device="cuda"):
    model = resnet152Places365.resnet152_places365
    model.load_state_dict(torch.load("./src/resnet152Places365/resnet152Places365.pth"))
    model.eval()
    model.to(device)
    return model


def preprocess_image(image: Image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)


def get_embeddings(images: torch.Tensor, layer_from_end=2, device="cuda") -> torch.Tensor:
    model = load_model()
    activation_extractor = ActivationExtractor(model, layer_from_end=layer_from_end)
    images = images.to(device)
    embeddings = activation_extractor.get_activation(images).to(device)
    return embeddings


if __name__ == "__main__":
    print("Starting the embedding extraction process")

    # Hyperparameters
    batch_size = 16
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataset and dataloader
    dataset = ImageDataset(image_folder="/data/images/", sample=764471)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    save_embeddings = embedding_saver_factory()
    print("Connected to the database")

    counter = 0
    for img_hashes, images in tqdm(dataloader, desc="Extracting Embeddings"):
        try:
            embeddings = get_embeddings(images, 2, device)
            save_embeddings(img_hashes, embeddings)
            counter += len(img_hashes)
            if counter % 5000 == 0:
                print(f"Extracted embeddings for {counter} images")
        except Exception as e:
            print(f"Failed to extract embeddings for batch with error {e}")
            continue
