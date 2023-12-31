from Mongo.mongo_connection import *
from Mongo.mongo_query import *
from Mongo.push_data_to_mongodb import *
import utils
from utils import initialise_project
from Image_color_moment import color_moments
from get_hist_og import histogram_of_oriented_gradients
from resnet_50 import resnet_features
from tqdm import tqdm

color = color_moments()
hog = histogram_of_oriented_gradients()
resnet = resnet_features()

class Task0a():

    def __init__(self) -> None:
        self.feature_dict_avgpool = {}
        self.feature_dict_layer3 = {}
        self.feature_dict_fc_layer = {}
        self.feature_dict_color_moments = {}
        self.features_dict_hog = {}
        self.softmax = {}
        self.feature_list = []
        self.dataset, self.labelled_images = initialise_project()
        pass

    def runTask0a(self):
        print("*"*25 + " Task 0a "+ "*"*25)
        print("This task will first delete all records from all the collections.\n Please enter '0' if you want to exit or any else value to continue\n")
        d = int(input())

        if d != 0:
            self.delete_all_records()
            print("Please enter the choice below:\n1 - Load from pickle files\n2 - Compute feature vectors and load\n ")
            n = int(input())
            if n != 2:
                if n != 1:
                    print("Invalid choice. Going with default input")
                
                try:
                    self.load_from_pickle_files()
                except Exception as e:
                    print("The pickle files could not be located:")
                    self.load_by_computing()
                
            else: 
                self.load_by_computing()

            print("The image feature descriptors have been stored in Database")
        
        else:
            print("Exiting task 0a.......")

    def custom_feature_extraction(self, i, image):

        color_moments_feature = color.color_moments_fn(image)
        hog_features = hog.compute_hog(image)

        # To compute we need to re inintialize the object again - reason for defining here
        resnet = resnet_features()

        # To run the resnet50 model
        run_model = resnet.run_model(image)
        avgpool_features = resnet.resnet_avgpool()
        layer3_features = resnet.resnet_layer3()
        fc_layer_features = resnet.resnet_fc_layer()
        softmax_features = resnet.apply_softmax()

        
        # Saving the outputs of all the layers in the resnet model
        self.feature_dict_avgpool[i] = avgpool_features
        self.feature_dict_layer3[i] = layer3_features
        self.feature_dict_fc_layer[i] = fc_layer_features
        self.softmax[i]= softmax_features

        # Saving the outputs of HOG and Color moments features
        self.feature_dict_color_moments[i] = color_moments_feature
        self.features_dict_hog[i] = hog_features
            
        return avgpool_features
    
    # Creates a dict in form of the document stored in collection
    def add_to_map(self, data, i):
        _, _, label = utils.img_label_and_named_label_for_query_int(self.dataset, i)
        data = data.tolist()
        map = {
            "imageID": i,
            "label": label or "unknown",
            "feature_descriptor": data
            }
        return map
    
    # Inserts the data into selected collection
    def add_to_database(self, collection_name, data):
        db = mongo_connection.get_database()
        collection = db[collection_name]
        result = collection.insert_one(data)
        print("Inserted document ID:", result.inserted_id)

    # Truncates the given collection
    def empty_collections(self, collection_name):
        db = mongo_connection.get_database()
        collection = db[collection_name]
        collection.delete_many({})

    # stores all the vectors to the database
    def delete_all_records(self):

        self.empty_collections('color_moment')
        self.empty_collections('hog')
        self.empty_collections('avgpool')
        self.empty_collections('layer3')
        self.empty_collections('fc_layer')
        self.empty_collections('resnet_final')

    def load_from_pickle_files(self):
        def onlyEvenData(data):
            # map = {
            #     "imageID": imageid,
            #     "label": labelled_data[imageid] or "unknown",
            #     "feature_descriptor": file_data[imageid].tolist()
            # }
            reduced_data = []
            for i in data:
                if i['imageID'] % 2 == 0: reduced_data.append(i)
            return reduced_data
        
    
        print("Starting loading form pickle files")
        labelled_data = utils.get_image_categories()
        # avgpool vectors
        data = combine_data("avgpool_vectors.pkl", labelled_data)
        upsert_data("avgpool", onlyEvenData(data))
        # Color_moments_vectors.pkl
        data = combine_data("Color_moments_vectors.pkl", labelled_data)
        upsert_data("color_moment", onlyEvenData(data))
        # fc_layer_vectors.pkl
        data = combine_data("fc_layer_vectors.pkl", labelled_data)
        upsert_data("fc_layer", onlyEvenData(data))
        # HOG_vectors.pkl
        data = combine_data("HOG_vectors.pkl", labelled_data)
        upsert_data("hog", onlyEvenData(data))
        # layer3_vectors.pkl
        data = combine_data("layer3_vectors.pkl", labelled_data)
        upsert_data("layer3", onlyEvenData(data))
        # resnet_vectors.pkl
        data = combine_data("resnet_vectors.pkl", labelled_data)
        upsert_data("resnet_final", onlyEvenData(data))
        
        # Compute and load into database
    def load_by_computing(self):
            
        total_image = len(self.dataset)
        for i in tqdm(range(0, total_image, 2), desc= 'Running feature extraction on all models'):
            image, _ = self.dataset[i]
            
            self.custom_feature_extraction(i, image)

            map = self.add_to_map(self.feature_dict_color_moments[i], i)
            self.add_to_database('color_moment', map)
            map = self.add_to_map(self.features_dict_hog[i], i)
            self.add_to_database('hog', map)
            map = self.add_to_map(self.feature_dict_avgpool[i], i)
            self.add_to_database('avgpool', map)
            map = self.add_to_map(self.feature_dict_layer3[i], i)
            self.add_to_database('layer3', map)
            map = self.add_to_map(self.feature_dict_fc_layer[i], i)
            self.add_to_database('fc_layer', map)
            map = self.add_to_map(self.softmax[i], i)
            self.add_to_database('resnet_final', map)

            
        print("The image feature descriptors have been stored in Database")

if __name__ == '__main__':
    task = Task0a()
    task.runTask0a()
    