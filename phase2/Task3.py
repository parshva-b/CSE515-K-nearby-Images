import torch
import utils
import dimension_reduction as dr
import numpy as np
import distances


class task3:
    def __init__(self) -> None:
        pass
    def k_latent_semantics(self):
        """This programs is to get the input from the user and calculate the LS1 using the chosen latent semnatics"""
        print("*"*25 + " Task 3 "+ "*"*25)
        print("Please select from below mentioned options")
        data, feature, dbName = utils.get_user_selected_feature_model()
        k = utils.get_user_input_k()
        ls_option = utils.get_user_selected_dim_reduction()
        
        # V -Input matrix with all the even images
        V = np.array(data)
        V = utils.convert_higher_dims_to_2d(V)
        
        print(V.shape)
        match ls_option:
            case 1: W, _, _ = dr.svd(V, k)
            case 2: W, _ = dr.nmf_als(V, k)
            case 3: W = dr.lda(V, k)
            case 4: W = dr.K_means(k, V)
            case default: print('No matching input was selected')
        
        path =  str("./LatentSemantics/LS1/LS1_" + utils.feature_model[feature]) + "_" + str(utils.latent_semantics[ls_option]) + "_" + str(k) + ".pkl"
        torch.save(W, path)
        print("Output file is saved with name - " + path)
        data = W

        # Printing the ImageID weight pairs in decreasing order
        utils.print_decreasing_weights(data, "ImageID")
        print("Exiting Task3 .............")

if __name__ == "__main__":
    temp = task3()
    temp.k_latent_semantics()
    # data = torch.load("./LatentSemantics/LS1/LS1_avgpool_layer_K_means_10.pkl")
    
    # df = pd.DataFrame()
    # k = 10
    # for val in range(k):
    #     ls = data[: ,val]
    #     indexed_list = list(enumerate(ls))
    #     sorted_list = sorted(indexed_list, key=lambda x: x[1])
    #     sorted_list.reverse()
    #     df["LS"+str(val+1) + "  ImageID, Weight"] = sorted_list
    #     output_list = sorted_list[- (k+1):].copy()
    #     output_list.reverse()
    #     print("Column : " + str(val))
    #     print("Output Format - ImageID, Weight")
    #     print(output_list)
    #     print("\n"*1)
    # print("Exiting Task3 .............")
    
