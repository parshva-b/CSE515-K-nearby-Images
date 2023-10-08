import pymongo
from pymongo import MongoClient

mongo_host = "localhost"  # or the IP address of your MongoDB server
mongo_port = 27017         # the default MongoDB port
db_name = "feature_descriptors"  # replace with your database name

def get_client():
    try:
        client = MongoClient(host=mongo_host, port=mongo_port)
        return client
    except pymongo.errors.ConnectionFailure as e:
        print("Failed to connect to MongoDB:", e)

def get_database():
    client = get_client()
    db = client[db_name]
    return db

def create_database():
    try:
        client = get_client()
        db = client[db_name]
        print("Connected to MongoDB and switched to database:", db_name)
    except pymongo.errors.ConnectionFailure as e:
        print("Failed to connect to MongoDB:", e)

def insert_document():
    try:
        client = get_client()
        db = client[db_name]
        print("Connected to MongoDB and switched to database:", db_name)
        collection = db["avgpool"]
        data = {"key": "value"}
        result = collection.insert_one(data)
        print("Inserted document ID:", result.inserted_id)

        # Example: Query data from a collection
        query_result = collection.find_one({"key": "value"})
        print("Query result:", query_result)
    except pymongo.errors.ConnectionFailure as e:
        print("Failed to connect to MongoDB:", e)    
