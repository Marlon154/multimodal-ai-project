import torch
import torchvision.transforms as transforms
from PIL import Image
import torchfile


def load_t7_model(t7_model_path):
    # Load the .t7 model
    t7_model = torchfile.load(t7_model_path)

    # Convert the .t7 model to PyTorch (assuming the Torch7 model can be loaded directly)
    # Here, we should normally convert layers manually, but assuming it is directly compatible
    model = torch.jit.load(t7_model_path)

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


def main(t7_model_path, image_path):
    model = load_t7_model(t7_model_path)

    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Run the prediction
    outputs = predict(model, image_tensor)

    # Print the outputs
    print(outputs)


if __name__ == "__main__":
    t7_model_path = "path_to_your_resnet152.t7"
    image_path = "path_to_your_image.jpg"
    main(t7_model_path, image_path)
