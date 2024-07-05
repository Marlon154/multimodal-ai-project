import json
import os
import shutil

from pymongo import MongoClient


def connect():
    client = MongoClient("mongodb://root:secure_pw@localhost:27017/")
    return client


def get_db(client, db):
    return client[db]


def get_collection(db, collection):
    return db[collection]


client = connect()

nytimes_db_sample = get_db(client, "nytimes_sample")
images_col_sample = get_collection(nytimes_db_sample, "images")

image_ids = [img["_id"] for img in images_col_sample.find()]

image_folder = "../../input_data/uncompressed/images_processed/"

destination_folder = "../../dump/sample_images/"

for id in image_ids:
    image_extension = ".jpg"
    image_path = os.path.join(image_folder, id + image_extension)

    destination_path = os.path.join(destination_folder, id + image_extension)
    shutil.copy2(image_path, destination_path)
