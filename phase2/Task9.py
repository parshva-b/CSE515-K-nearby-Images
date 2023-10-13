# Task 9 
# (a) a label l, (b) a user selected latent semantics, and (c) positive integer k, 
# identifies and lists k most likely matching labels, along with their scores, under the selected latent space.
import utils
import label_vectors
import os
import numpy as np
import torch
import dimension_reduction
from sklearn.metrics.pairwise import cosine_similarity

class task9:
    def __init__(self) -> None:
        pass
    
    def menu2(self):
        # # Select a label
        labels = utils.print_labels()
        label_num = utils.int_input()
        selected_label = labels[label_num] # 0 index
        selected_label = selected_label.lower()
        print("\t selected label" , selected_label)
        
        # Show latent_features and give option to select [LS1, LS2, LS3, etc.]
        latent_feature = utils.get_user_input_latent_feature()

        print(f"selected latent feature is = {latent_feature}")
        path = "./LatentSemantics/"+latent_feature

        


        # show available files
        print("Available Latent Semantics : ")
        file_names = os.listdir(path)
        for i in range(len(file_names)):
            file_name = file_names[i]
            if file_name != ".gitkeep":
                print(f"\t {i} : {file_name}")
        
        # print("\n\t Select Latent Semantic")
        semantic_option = utils.int_input()
        selected_latent_semantic = file_names[semantic_option]
        print("selected_latent_semantic ",  selected_latent_semantic)
        # find which feature space it is using
        feature_models = utils.feature_model
        feature_models_values = list(feature_models.values())
        
        for feature_space in feature_models_values:
            feature_space = feature_space.strip().lower()
            selected_latent = selected_latent_semantic.lower()
            if feature_space in selected_latent:
                detected_feature_space = feature_space
                break  # Exit the loop as soon as a match is found
        print(f"detected feature space = {detected_feature_space}")


        ## we have detected_feature_space and selected_label
        ## We will now query all the feature vectors
        _, labelled_images = utils.initialise_project()
        model_space = torch.load('fc_layer_vectors.pkl')
        labels_feature_vector = label_vectors.label_fv_kmediods(selected_label, labelled_images, model_space)
        # print(labels_feature_vector)
        # find image similar to labels_feature_vector
        similar_image = label_vectors.label_image_distance_using_cosine(len(model_space), labels_feature_vector, model_space, 1)
        print(similar_image[0][0])
        check_index = int(similar_image[0][0] / 2)
        print(check_index)
        latent_semantics = torch.load(path+"/"+selected_latent_semantic)
        semantic_vector = latent_semantics[check_index]

        similarities = cosine_similarity(latent_semantics, [semantic_vector])
        distances = 1 - similarities

        sorted_indices = np.argsort(distances, axis=0).flatten()
        print(sorted_indices)
        labels_of_each_indices = []
        image_labels = utils.get_image_categories()
        for i in sorted_indices:
            label = image_labels[i*2]
            labels_of_each_indices.append(label)

        seen = set()
        unique_elements = []
        for item in labels_of_each_indices:
            if item not in seen:
                unique_elements.append(item)
                seen.add(item)
                if len(unique_elements) == 5:
                    break
        print(unique_elements)
        return

if __name__ == "__main__":
    temp = task9()
    # temp.menu()
    temp.menu2()