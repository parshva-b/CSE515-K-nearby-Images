from utils import *
from label_vectors import *
import distances 
import dimension_reduction as dr
class Task5:
    def __init__(self) -> None:
        self.dataset, self.labelled_images = initialise_project()

    def label_label_similarity(self):
        '''Takes the input from user for feature model selection. 
        Returns the label-label similarity matrix'''
        labelled_feature_vectors, modelSpace, option = create_labelled_feature_vectors(self.labelled_images)
        labelled_feature_list = []
        for index in range(len(self.dataset.categories)):
            label_name = name_for_label_index(self.dataset, index)
            labelled_feature_list.append(labelled_feature_vectors[label_name])
        
        labelled_feature_descriptor = np.array(labelled_feature_list)
        total_labels = labelled_feature_descriptor.shape[0]
        labelled_feature_descriptor = labelled_feature_descriptor.reshape(total_labels, -1)

        label_label_similarity_matrix = np.zeros((total_labels,total_labels))

        for i in range(total_labels):
            for j in range(total_labels):
                if option == 6:
                    label_label_similarity_matrix[i][j] = distances.kl_divergence(labelled_feature_descriptor[i], labelled_feature_descriptor[j])
                else:
                    label_label_similarity_matrix[i][j] = distances.cosine_similarity(labelled_feature_descriptor[i], labelled_feature_descriptor[j])
        path =  str("./LatentSemantics/LS4/label_label_similarity_matrix_" + feature_model[option]) + ".pkl"
        torch.save(label_label_similarity_matrix, path)
        print("Output file for label-label similarity matrix is saved with name - " + path)
        
        return label_label_similarity_matrix, option
        # print(labelled_feature_vectors.shape)

    def label_dimensionality_reduction(self, data):
        '''Takes the input dimensioanlity reduction technique and k from user 
        Returns the reduced label-label similarity matrix, option selected by user and k'''
        k = get_user_input_k()
        option = get_user_selected_dim_reduction()

        match option:
            case 1:
                reduced_data, Sigma, VT = dr.svd(data, k, True)
            case 2:
                reduced_data, H = dr.nmf_als(data, k, )
            case 3:
                reduced_data = dr.lda(data, k)
            case 4: 
                reduced_data = dr.K_means(k, data)
            case default:
                return print('No matching input was selected')
            
        return reduced_data, option, k

    def save_to_path(self, W, feature, ls_option, k):
        '''Saves the reduced label-label similarity matrix'''
        path =  str("./LatentSemantics/LS3/LS3_" + feature_model[feature]) + "_" + str(latent_semantics[ls_option]) + "_" + str(k) + ".pkl"
        torch.save(W, path)
        print("Output file is saved with name - " + path)

if __name__ == '__main__':
    task5 = Task5()
    print("*"*25 + " Task 5 "+ "*"*25)
    print("Please select from below mentioned options")
    label_label_similarity_matrix, feature_option = task5.label_label_similarity()
    reduced_matrix, dr_option, k = task5.label_dimensionality_reduction(label_label_similarity_matrix)
    task5.save_to_path(reduced_matrix, feature_option, dr_option, k)
    print_decreasing_weights(reduced_matrix, "Labels")
    print("Exiting Task5 .............")

        
    