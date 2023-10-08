import mongo_connection
from bson.json_util import dumps

def get_collection(collection_name):
    db = mongo_connection.get_database()
    collection = db[collection_name]
    return collection

def convert_to_json(cursor):
    list_cur = list(cursor)
    json_data = dumps(list_cur, indent = 2)
    return json_data

def query(collection_name, imageID):
    collection = get_collection(collection_name)
    cursor = collection.find({"imageID": imageID})
    json_data = convert_to_json(cursor)
    return json_data

def query_all(collection_name):
    collection = get_collection(collection_name)
    cursor = collection.find()
    json_data = convert_to_json(cursor)
    return json_data