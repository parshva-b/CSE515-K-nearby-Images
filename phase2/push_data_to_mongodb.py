import mongo_connection
import pickle
import main
import torch

def read_file(filename):
    featureDesc = torch.load(filename)
    return featureDesc

def get_data_to_store(file_data, labelled_data):
    data = []
    for imageid in file_data:
        map = {
            "imageID": imageid,
            "label": labelled_data[imageid] or "unknown",
            "feature_descriptor": file_data[imageid].tolist()
        }
        data.append(map)  
    return data

def upsert_data(collection_name, data):
    db = mongo_connection.get_database()
    collection = db[collection_name]
    for d in data:
        result = collection.insert_one(d)
        print("Inserted document ID:", result.inserted_id)

def combine_data(filename, labelled_data):
    file_data = read_file(filename)
    data = get_data_to_store(file_data, labelled_data)
    return data

def process():
    labelled_data = main.get_labelled_images()
    # avgpool vectors
    data = combine_data("CSE515_PKL_Files/avgpool_vectors.pkl", labelled_data)
    upsert_data("avgpool", data)
    # Color_moments_vectors.pkl
    data = combine_data("CSE515_PKL_Files/Color_moments_vectors.pkl", labelled_data)
    upsert_data("color_moment", data)
    # fc_layer_vectors.pkl
    data = combine_data("CSE515_PKL_Files/fc_layer_vectors.pkl", labelled_data)
    upsert_data("fc_layer", data)
    # HOG_vectors.pkl
    data = combine_data("CSE515_PKL_Files/HOG_vectors.pkl", labelled_data)
    upsert_data("hog", data)
    # layer3_vectors.pkl
    data = combine_data("CSE515_PKL_Files/layer3_vectors.pkl", labelled_data)
    upsert_data("layer3", data)
    # resnet_vectors.pkl
    data = combine_data("CSE515_PKL_Files/resnet_vectors.pkl", labelled_data)
    upsert_data("resnet_final", data)
    
process()
