import torch
import utils
import dimension_reduction as dr
import numpy as np
import distances
import Mongo.mongo_query_np  as monogo_query
from sortedcollections import OrderedSet


class task4:
    def __init__(self) -> None:
        pass
    def LS2_cp_decompose(self):
        """This programs is to get the input from the user and calculate the LS2 using CP decomoposition"""

        print("*"*25 + " Task 4 "+ "*"*25)
        print("Please select from below mentioned options")
        data, option, dbName = utils.get_user_selected_feature_model()
        k = utils.get_user_input_k()
        data = monogo_query.get_entire_collection(dbName)


        data = self.get_tensor(data)
        print(len(data))
        factors = dr.cp_decompose(data,k)
        print(factors[0])
        
        # Getting the label weight pairs
        label_weights = factors[1][2]
        # image_weights = factors[1][0]
        # feature_weights = factors[1][1]

        # Printing the ImageID weight pairs in decreasing order
        utils.print_decreasing_weights(label_weights, "Label")
        
        path =  str("./LatentSemantics/LS2/LS2_" + dbName) + "_CP_decompose_" + str(k) + ".pkl"
        torch.save(factors, path)
        print("Output file is saved with name - " + path)

        print("Exiting Task4 .............")


    def get_tensor(self, data) :
        
        #Get data in list of tuples form each tuple containing - Imageid - feature_descriptor - label 
        required_data = [ (entry["imageID"],np.array(entry["feature_descriptor"]).flatten(), entry["label"])  for entry in data ]
        
        #Get number of images, labels and length of the feature_descriptor
        images_len = len( OrderedSet([x[0] for x in required_data]))
        features_len = len(required_data[0][1])
        label_len = len( OrderedSet([x[2] for x in required_data]))
        
        label_id_mapping = list(OrderedSet([x[2] for x in required_data]))
        
        #Create empty tensor
        model = np.zeros((images_len,features_len,label_len))
        print(model.shape)

        #Tensor creation and assign values
        i = 0
        for entry in required_data :
            
            image_id, feature, label_id = entry[0], entry[1], label_id_mapping.index(entry[2])
            model[i, : , label_id ] = feature
            i += 1

        # Tried adding random values and ones in the empy space - Results detirieted after that 
        # random_values = np.random.rand(*model.shape)  
        # final = np.where(model == 0, random_values, model)
        # final = np.where(model == 0, 1, model)

        return model

if __name__ == "__main__":
    temp = task4()
    temp.LS2_cp_decompose()
    # dataset, _ = utils.initialise_project()
    # print(utils.name_for_label_index(dataset=dataset, index=94 ))

