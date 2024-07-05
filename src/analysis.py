from pymongo import MongoClient, collection

def connect():
    client = MongoClient("mongodb://root:secure_pw@mongo_db:27017/")
    return client


if __name__ == "main":

    # Connect to database
    client = connect()
    db = client["nytimes"]
    article_table = self.db["articless"]