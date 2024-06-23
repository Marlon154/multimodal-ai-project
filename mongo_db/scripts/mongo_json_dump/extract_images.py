import json
import os
import shutil

json_folder = "./sample_json/"

image_folder = "../uncompressed/images_processed/"

destination_folder = "./sample_images/"

for filename in os.listdir(json_folder):
    json_path = os.path.join(json_folder, filename)

    with open(json_path, "r") as f:
        data = json.load(f)

    sections = data["parsed_section"]
    hashes = []
    for section in sections:
        if section["type"] == "caption":
            hashes.append(section["hash"])

    for hash in hashes:
        image_extension = ".jpg"
        image_path = os.path.join(image_folder, hash + image_extension)

        destination_path = os.path.join(destination_folder, hash + image_extension)
        shutil.copy2(image_path, destination_path)
