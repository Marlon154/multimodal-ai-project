import json

from pymongo import MongoClient


def connect():
    client = MongoClient("mongodb://root:secure_pw@localhost:27017/")
    return client


def get_db(client, db):
    return client[db]


def get_collection(db, collection):
    return db[collection]


client = connect()
nytimes = get_db(client, "nytimes")
articles = get_collection(nytimes, "articles")
images = get_collection(nytimes, "images")
objects = get_collection(nytimes, "objects")


sample_article = articles.find_one()
sample_image = images.find_one()
sample_object = objects.find_one()

sample_articles = articles.find().limit(1000)


def save_to_json(document):
    name = f"{document['_id']}.json"
    with open(f"sample_json/{name}", "w") as file:
        json.dump(document, file, indent=4)


for sample in sample_articles:
    sample["pub_date"] = sample["pub_date"].isoformat()
    save_to_json(sample)
