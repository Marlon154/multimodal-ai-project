import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from pymongo import MongoClient, collection
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# DO NOT TOUCH
from resnet152Places365 import resnet152Places365


class ActivationExtractor:
    """
    Custom class using pytorch hooks to extract the activations from an arbitrary layer
    Args:
        model: instance of torch.nn.Sequential defining the model
        layer_from_end: integer specifying the layer whos activations should be extracted. if equal to 1 returns activations of the last layer
    """

    def __init__(self, model: torch.nn.Sequential, layer_from_end):
        self.model = model
        model.eval()
        self.layer_from_end = layer_from_end
        self.hook_handle = None
        self.activation = None

        self.layer_index = len(list(model.children())) - layer_from_end
        if self.layer_index < 1:
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
    """
    Torch Dataset wrapper class around the images of NYT800K.
    Implements all methods required by torch.utils.data.Dataset
    """

    def __init__(self, client, image_folder="/app/data/nytimes/images_processed/", image_extension=".jpg", sample=0):
        self.image_folder = image_folder
        self.image_extension = image_extension
        self.db = client["nytimes"]
        self.image_table = self.db["images"]
        docs = list(self.image_table.find().limit(sample))
        self.document_ids = []
        for doc in docs:
            img_path = os.path.join(self.image_folder, doc["_id"] + self.image_extension)
            if os.path.exists(img_path):
                self.document_ids.append(doc["_id"])
        print(f"checked images: {len(self.document_ids)}")

    def __len__(self):
        return len(self.document_ids)

    def __getitem__(self, idx):
        id = self.document_ids[idx]
        img_path = os.path.join(self.image_folder, id + self.image_extension)
        image = Image.open(img_path).convert("RGB")
        image = preprocess_image(image)
        return id, image


def load_model(device="cuda"):
    '''
        Instanciates the pretrained ResNet, trained on the Places365 dataset in eval mode and on the target device
        Returns the model instance
    '''
    model = resnet152Places365.resnet152_places365
    model.load_state_dict(torch.load("./src/resnet152Places365/resnet152Places365.pth"))
    model.eval()
    model.to(device)
    return model


def preprocess_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(image)


if __name__ == "__main__":
    '''
        Generate embeddings for all images in the dataset. 
        Embeddings are saved into a new collection in the original database
        The embeddings are the activation of the second to last layer of the Places365 Resnet model
    '''
    print("Starting the embedding extraction process")

    client = connect()
    db = client["nytimes"]

    # check/create place_embeddings
    collection_list = db.list_collection_names()
    collection_name = "place_embeddings"
    if not collection_name in collection_list:
        db.create_collection(collection_name)
    embedding_table = db["place_embeddings"]
    print("Connected to the database")

    # remove all documents from the collection to make sure there are no duplicates
    embedding_table.delete_many({})

    # Hyperparameters
    batch_size = 1024
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = ImageDataset(client=client, image_folder="/data/images/", sample=764471)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    def store_embeddings(keys, embeddings: torch.Tensor):
        embeddings = embeddings.cpu().numpy().tolist()
        records = [{key: embedding} for key, embedding in zip(keys, embeddings)]
        embedding_table.insert_many(records)

    layer_from_end = 2

    print(f"Loading Model")
    model = load_model()
    activation_extractor = ActivationExtractor(model, layer_from_end=layer_from_end)

    print(f"Starting scene embedding")
    counter = 0
    for img_hashes, images in tqdm(dataloader, desc="Extracting Embeddings"):
        images = images.to(device)
        embeddings = activation_extractor.get_activation(images)
        store_embeddings(img_hashes, embeddings)
        counter += len(img_hashes)
        if counter % 1000 == 0:
            print(f"Extracted embeddings for {counter} images")
