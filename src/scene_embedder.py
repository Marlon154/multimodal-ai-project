import torch
import torchvision.transforms as transforms
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from PIL import Image
import torchfile
from pymongo import MongoClient
from PIL import Image
import os

# DO NOT TOUCH
from resnet152Places365 import resnet152Places365


def connect():
    client = MongoClient("mongodb://root:secure_pw@localhost:27017/")
    return client


def get_imgages(
    db="nytimes",
    image_folder="./../input_data/uncompressed/images_processed/",
    image_extension=".jpg",
    sample=None,
):
    # print("get_images...")
    client = connect()

    # print("Getting correct collection...")
    db = client["nytimes_sample"]
    image_table = db["images"]

    # print("Finding images...")
    ids = []
    # if sample
    documents = image_table.find().limit(sample)
    for doc in documents:
        ids.append(os.path.join(image_folder, doc["_id"] + image_extension))

    # print(f"{ids=}")
    return ids


def load_model():
    model = resnet152Places365.resnet152_places365
    model.load_state_dict(torch.load("./src/resnet152Places365/resnet152Places365.pth"))
    model.eval()
    return model


def preprocess_image(image_path):
    # transform for image
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path)

    # Apply the transformation
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    return image


def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    return outputs


def main(image_path):
    model = load_model()

    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Run the prediction
    outputs = predict(model, image_tensor)

    # Print the outputs
    print(outputs)


if __name__ == "__main__":
    # image_path = get_imgages(
    #     image_folder="./data/nytimes/sample/sample_images/", sample=1
    # )[0]
    load_model()
    # main(image_path)
