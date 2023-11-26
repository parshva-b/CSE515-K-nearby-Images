import utils
import distances as d
import numpy as np
from Mongo import mongo_query_np
from resnet_50 import *
from dimension_reduction import *
import os
import pickle
import math
import json
import warnings

warnings.filterwarnings("ignore")
resnet = resnet_features()

class Task4a:
    def __init__(self) -> None:
        self.dataset, self.labelled_images = initialise_project()
        data = mongo_query_np.get_all_feature_descriptor("fc_layer")
        file_name = 'NNMF_reduced_256.pkl'
        self.data_matrix = []
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                self.data_matrix = pickle.load(file)
        else:
            self.data_matrix, _ = nmf_als(data, 256)
            with open(file_name, 'wb') as file:
                    pickle.dump(self.data_matrix, file)

    def runTask4a(self):
        print("*"*25 + " Task 4a "+ "*"*25)
        print("Please enter the number of hashes:")
        num_hashes = int(utils.int_input(10))
        print("Please enter the number of layers:")
        num_layers = int(utils.int_input(10))

        return self.LSH(num_hashes, num_layers)

    
    def LSH(self, num_hashes, num_layers):
        print("Using the ResNet50 FC Layer visual model reducing the dimensions to 256")

        w = (num_layers * num_hashes) * int(math.sqrt(len(self.data_matrix))) / 2
        hyperplanes = [np.random.randn(num_hashes, self.data_matrix.shape[0], self.data_matrix.shape[1]) for _ in range(num_layers)]
        random_projections = [np.zeros((num_hashes, self.data_matrix.shape[0])) for _ in range(num_layers)]

        for layer in range(len(random_projections)):
            for hash_function in range(len(random_projections[layer])):
                for i in range(len(random_projections[layer][hash_function])):
                    random_projections[layer][hash_function][i] = np.dot(hyperplanes[layer][hash_function][i], self.data_matrix[i])

        divisor = []
        for i in random_projections:
            length = np.ptp(i, axis=1)
            divisor.append(length/w)


        neighbouring_index = [[[0 for _ in range(len(random_projections[0]))] for _ in range(len(random_projections))] 
                    for _ in range(len(random_projections[0][0]))]

        for layer in range(len(random_projections)):
            for hash_function in range(len(random_projections[layer])):
                for i in range(len(random_projections[layer][hash_function])):
                    neighbouring_index[i][layer][hash_function] = random_projections[layer][hash_function][i] // divisor[layer][hash_function]
        print("\nLSH index structure has been created in memory\n")
        return neighbouring_index
    

class Task4b:
    def __init__(self) -> None:
        self.dataset, self.labelled_images = initialise_project()
        data = mongo_query_np.get_all_feature_descriptor("fc_layer")
        file_name = 'NNMF_reduced_256.pkl'
        self.data_matrix = []
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                self.data_matrix = pickle.load(file)
        else:
            self.data_matrix, _ = nmf_als(data, 256)
            with open(file_name, 'wb') as file:
                    pickle.dump(self.data_matrix, file)

    def runTask4b(self, hash_codes):
        print("*"*25 + " Task 4b "+ "*"*25)

        imageID = utils.get_user_input_image_id()
        

        if imageID % 2 != 0:
            img, _ = self.dataset[imageID]
            run_model = resnet.run_model(img)
            query_vector = resnet.resnet_fc_layer()
            imageID = self.get_closest_image_id(query_vector)

        neighbouring_index = self.approx_images(imageID, hash_codes)
        print("Number of considered images: ", len(neighbouring_index))
        print("Indices of images: ", neighbouring_index)
        print("Please enter the number of relevant images (T):")
        k = utils.int_input(10)
        similar_images = self.knn(imageID, k, neighbouring_index)
        utils.display_k_images_subplots(self.dataset, similar_images, f"{k} most relevant images using LSH index structure")


    def get_closest_image_id (self, input_image_vector):
        """This function Quries that DB and gets the entire feature space for FC, then it finds the closest EVEN
           image id using cosine distance in a loop
            Input - input_image_vector
            Output - Closest EVEN image ID
            """

        db_data = mongo_query_np.get_all_feature_descriptor(utils.feature_model[5])
        output = self.distance_function_for_images(input_image_vector, db_data, 1)
        closest_image_id = output[0][0]
        return closest_image_id
    
    def distance_function_for_images(self, query_vector, data, k):
        """Runs the distance function in loop gets you the K top images"""
        distances = []
        for i in range(len(data)):
            distances.append(d.cosine_similarity(query_vector.flatten(), data[i]))
        indexed_list = list(enumerate(distances))

        # Sorting the the list 
        sorted_list = sorted(indexed_list, key=lambda x: x[1])

        output_list = sorted_list[- (k+1):].copy()
        output_list.reverse()
        
        output_list = [(x * 2, y) for x, y in output_list]

        return output_list
    
    def approx_images(self, imageID, hash_codes):

        q = imageID//2

        neighbouring_index = set()

        for i in range(len(hash_codes)):
            for j in range(len(hash_codes[i])):
                for m, n in zip(hash_codes[q][j], hash_codes[i][j]):
                    if abs(m - n) == 0:
                        neighbouring_index.add(i*2)

        data = {
            'query_image': imageID,
            'neighbour_images': list(neighbouring_index)
        }

        json_data = json.dumps(data, indent=2)

        with open('4b_output.json', 'w') as json_file:
            json_file.write(json_data)

        return neighbouring_index

    def knn(self, img, k, neighbouring_index):
    
        lsh_set = []
        for i in neighbouring_index:
            lsh_set.append(i//2)

        img = img//2
        distances = []
        
        for i, data_point in enumerate(self.data_matrix):
            if i in lsh_set and i != img:
                distance = d.cosine_distance(self.data_matrix[img], data_point)
                distances.append((i, distance))
            

        distances.sort(key=lambda x: x[1])
        similar_images = [(index*2, dist) for index, dist in distances[:k]]
        return similar_images
    

if __name__ == '__main__':
    task4a = Task4a()
    hash_codes = task4a.runTask4a()

    task4b = Task4b()
    task4b.runTask4b(hash_codes)