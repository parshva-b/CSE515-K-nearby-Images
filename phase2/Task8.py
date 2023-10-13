import utils
import torch
import distances as ds
import numpy as np
import torchvision
from collections import defaultdict
from Mongo import mongo_query_np
from resnet_50 import *

resnet = resnet_features()

class Task8:
    def __init__(self) -> None:
        self.dataset, self.labelled_images = utils.initialise_project()

    def runTask8(self) -> None:

        imageID, img = utils.get_user_input_internalexternal_image()
        pathname, option = utils.get_user_input_latent_semantics()
        k = utils.get_user_input_k()
        feature_data = torch.load(pathname)

        if imageID % 2 !=0 and imageID != -1:
            img, _ = self.dataset[imageID]
            run_model = resnet.run_model(img)
            feature_descriptor = resnet.resnet_fc_layer()
            imageID = self.get_closest_image_id(feature_descriptor)
        elif imageID == -1:
            run_model = resnet.run_model(img)
            feature_descriptor = resnet.resnet_fc_layer()
            imageID = self.get_closest_image_id(feature_descriptor)

        if option == 1 or option == 4:
            result = self.distance_function_labels_from_images(feature_data[int(imageID/2)], feature_data, k)
        elif option == 2:
            feature_data = feature_data[1][2]
            _, label = self.dataset[imageID]
            result = self.distance_function_for_images_labels(feature_data[int(label)], feature_data, k)
            
        else:
            _, label = self.dataset[imageID]
            result = self.distance_function_for_images_labels(feature_data[int(label)], feature_data, k)
        
        print(result)
            

    def get_closest_image_id (self, input_image_vector):
        """This function Quries that DB and gets the entire feature space for FC, then it finds the closest EVEN
           image id using cosine distance in a loop
            Input - input_image_vector
            Output - Closest EVEN image ID
            """

        # Get the entire dataset to find the closest even image
        db_data = mongo_query_np.get_all_feature_descriptor(utils.feature_model[5])
        output = self.distance_function_for_images_images(input_image_vector, db_data, 1)
        closest_image_id = output[0][0]
        return closest_image_id
        


    def distance_function_labels_from_images(self, query_vector, data, k):
        """Runs the distance function in loop gets you the K top labels"""
        distances = []
        for i in range(len(data)):
            distances.append(ds.cosine_similarity(query_vector.flatten(), data[i]))
        indexed_list = list(enumerate(distances))

        sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse= True)
        output_list = set()
        result = []
        i = 0
        while len(output_list) != k:
            j = 2*(sorted_list[i][0])
            _, label = self.dataset[j]

            if label not in output_list:
                output_list.add(label)
                result.append((utils.name_for_label_index(self.dataset, label), sorted_list[i][1]))
            i += 1

        return result
    
    def distance_function_for_images_labels(self, query_vector, data, k):
        """Runs the distance function in loop gets you the K top images"""
        distances = []
        for i in range(len(data)):
            distances.append(ds.cosine_similarity(query_vector.flatten(), data[i]))
        indexed_list = list(enumerate(distances))

        # Sorting the the list 
        sorted_list = sorted(indexed_list, key=lambda x: x[1])

        output_list = sorted_list[- (k+1):].copy()
        output_list.reverse()
        output_list = [(utils.name_for_label_index(self.dataset, x), y) for x, y in output_list]

        return output_list

    def distance_function_for_images_images(self, query_vector, data, k):
        """Runs the distance function in loop gets you the K top images"""
        distances = []
        for i in range(len(data)):
            distances.append(ds.cosine_similarity(query_vector.flatten(), data[i]))
        indexed_list = list(enumerate(distances))

        # Sorting the the list 
        sorted_list = sorted(indexed_list, key=lambda x: x[1])

        output_list = sorted_list[- (k+1):].copy()
        output_list.reverse()
        
        output_list = [(x * 2, y) for x, y in output_list]

        return output_list
# /Users/suyashsutar99/Downloads/ImportantDoc.jpg
if __name__ == '__main__':
    task8 = Task8()
    task8.runTask8()