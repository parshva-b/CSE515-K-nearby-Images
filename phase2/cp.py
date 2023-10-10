from sortedcollections import OrderedSet
import numpy as np
import mongo_query
import json
from tensorly.decomposition import parafac
from tensorly.decomposition import parafac2

def get_data(feature_model) :
    
    '''
    Get data in form Image - feature_model - label form 
    '''
    
    data = mongo_query.query_all(feature_model)
    
    #convert to json
    json_data = json.loads(data)
    print("Data fetched from database")
    
    return json_data


def get_tensor(data) :
    
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
    for entry in required_data :
        
        image_id, feature, label_id = entry[0], entry[1], label_id_mapping.index(entry[2])
        model[image_id, : , label_id ] = feature

    return model
    
def cp( tensor : np.ndarray, rank : int, init="random") -> np.ndarray :

    '''
	Takes multimodal tensor a >= 3D numpy ndarray as input. Ex: imageID x feature descriptor X labelID
	Performs the cp decomposition using either parafac, parafac2 methods from tensorly library.
	Uses either SVD or random initialization to provide core and factor matrices.
	X [I1 x I2 x I3] = U1 [I1xR] U2 [I2xR] U3 [kxn]
    where R is the rank provided by the user.
	'''
    
    #Center the tensor - optional
    print("centering...")
    tensor = tensor - np.mean(tensor, axis=(0, 2), keepdims=True)
    
    print("Decomposing...")
    decomposition = parafac(tensor,rank,init=init, n_iter_max=100, verbose=1)

    return decomposition
    
def main():
    
    feature_model = "fc_layer"
    data = get_data(feature_model)
    tensor = get_tensor(data)
    decomposition = cp(tensor,10)
    
    
main()