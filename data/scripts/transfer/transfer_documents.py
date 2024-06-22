from pymongo import MongoClient


def connect():
    client = MongoClient("mongodb://root:secure_pw@localhost:27017/")
    return client


def get_db(client, db):
    return client[db]


def get_collection(db, collection):
    return db[collection]


client = connect()

nytimes_db = get_db(client, "nytimes")
articles_col = get_collection(nytimes_db, "articles")
images_col = get_collection(nytimes_db, "images")
objects_col = get_collection(nytimes_db, "objects")

nytimes_db_sample = get_db(client, "nytimes_sample")
articles_col_sample = get_collection(nytimes_db_sample, "articles")
images_col_sample = get_collection(nytimes_db_sample, "images")
objects_col_sample = get_collection(nytimes_db_sample, "objects")

# steps
# get all _id of articles
# add all images with captions > id in article_ids
# get all _id of images
# add all objects with captions > id in image_ids

# insert articles in sample db
insert_articles = articles_col.find().limit(1000)
articles_col_sample.insert_many(insert_articles)

articles = articles_col_sample.find()

# get image ids from articles in sample db
image_ids = []
for art in articles:
    paragraphs = art["parsed_section"]
    for par in paragraphs:
        # only parse section that have an image
        if par["type"] == "caption":
            image_ids.append(par["hash"])

# get images and add them to the sample db
insert_images = images_col.find({"_id": {"$in": image_ids}})
images_col_sample.insert_many(insert_images)

images = images_col_sample.find()

# get objects and add them to sample db
object_ids = [img["_id"] for img in images]
insert_objects = objects_col.find({"_id": {"$in": object_ids}})
objects_col_sample.insert_many(insert_objects)
