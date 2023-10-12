# Task 9 
# (a) a label l, (b) a user selected latent semantics, and (c) positive integer k, 
# identifies and lists k most likely matching labels, along with their scores, under the selected latent space.
import utils
import label_vectors
import os

class task9:
    def __init__(self) -> None:
        pass
    
    def menu2(self):
        # Select a label
        labels = utils.print_labels()
        label_num = utils.int_input()
        selected_label = labels[label_num-1] # 0 index
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
                print(f"\t {i+1} : {file_name}")
        
        print("\n\t Select Latent Semantic")
        semantic_option = utils.int_input()
        selected_latent_semantic = file_names[semantic_option-1]

        # find which feature space it is using
        feature_models = utils.feature_model
        feature_models_values = list(feature_models.values())
        
        for feature_space in feature_models_values:
            feature_space = feature_space.strip().lower()
            selected_latent = selected_latent_semantic.lower()
            if feature_space in selected_latent:
                detected_feature_space = feature_space
                break  # Exit the loop as soon as a match is found
        print(detected_feature_space)


        ## we have detected_feature_space and selected_label
        ## We will now query all the feature vectors
        _, labelled_images = utils.initialise_project()
        labelled_feature_vectors, _ = label_vectors.create_labelled_feature_vectors(labelled_images)
        
        print("\n\n Selected Labels Vector = #{}")

        return

    # def menu(self):
        utils.print_labels()
        # Add code to select Label here

        ####

        _, labelled_images = utils.initialise_project()
        ## give latent space selection option
        latent_feature = utils.get_user_input_latent_feature()
        print(f"selected latent feature is = {latent_feature}")
        path = "./LatentSemantics/"+latent_feature
        print("Available Latent Semantics : ")
        file_names = os.listdir(path)


        for i in range(len(file_names)):
            file_name = file_names[i]
            if file_name != ".gitkeep":
                print(f"{i+1} : {file_name}")

        feature_models = utils.feature_model
        feature_models_values = list(feature_models.values())
        print(feature_models_values)

        print('\n\nSelect Latent Semantic:')
        selected_latent_semantic_num = utils.int_input()
        selected_latent_semantic = file_names[selected_latent_semantic_num-1] # 0 indexing
        selected_latent = selected_latent_semantic.strip().lower()
        detected_feature_space = None

        # Iterate through the feature spaces
        for feature_space in feature_models_values:
            feature_space = feature_space.strip().lower()
            if feature_space in selected_latent:
                detected_feature_space = feature_space
                break  # Exit the loop as soon as a match is found

        # Check if a feature space was detected
        if detected_feature_space is not None:
            print(f"Detected feature space: {detected_feature_space}")
        else:
            print("No feature space detected in the selected_latent string.")

        labelled_feature_vectors, model_space = label_vectors.create_labelled_feature_vectors(labelled_images)
        print(labelled_feature_vectors.keys())

        # k = utils.get_user_input_k()
        return

if __name__ == "__main__":
    temp = task9()
    # temp.menu()
    temp.menu2()