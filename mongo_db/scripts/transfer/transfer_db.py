from pymongo import MongoClient


def connect():
    client = MongoClient("mongodb://root:secure_pw@localhost:27017/")
    return client


client = connect()


old_db = client["nytimes"]
new_db = client["nytimes_full"]

for collection_name in old_db.list_collection_names():
    print(f"Starting with collection: {collection_name}")
    old_collection = old_db[collection_name]
    new_collection = new_db[collection_name]

    documents = old_collection.find()
    print(f"Got all documents from: {collection_name}")
    new_collection.insert_many(documents)
    print(f"Inserted all documents from: {collection_name}")
