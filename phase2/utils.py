from typing import Tuple
import math
from collections import defaultdict

import torchvision
import torch
from matplotlib import pyplot
from torchvision import datasets
# from ordered_set import OrderedSet
from PIL import Image

feature_model  = {
    1 : "color_moments",
    2 : "hog",
    3 : "avgpool_layer",
    4 : "layer3",
    5 : "fc_layer"
}

latent_semantics = {
    1 : "SVD",
    2 : "NNMF",
    3 : "LDA",
    4 : "K_means"
}

def int_input(default_value: int = 99) -> int:
    try:
        inpu = int(input())
        return inpu
    except ValueError:
        print(f'No proper value was passed, Default value was used')
        return default_value


# for query image id, return label name for it
def name_for_label_index(dataset: torchvision.datasets.Caltech101, index: int) -> str:
    dataset_named_categories = dataset.categories
    return dataset_named_categories[index]

# for query image id, return (PIL image, label_id, label_name) 
# returns IndexError if index is not in range
def img_label_and_named_label_for_query_int(dataset: torchvision.datasets.Caltech101, 
                                            index: int) -> Tuple[any, int, str]:
    if(index > len(dataset) or index < 0): 
        print("Not proper images")
        return IndexError
    else:
        img, label_id = dataset[index]
        label_name = name_for_label_index(dataset, label_id)
        return (img, label_id, label_name)

def initialise_project():
    dataset = torchvision.datasets.Caltech101(root='./data', download=True, target_type='category')

    # this is going to be created once and passed throughout in all functions needed
    # dict: (label: string, list<imageIds: int>)
    # where key is either label index as int (eg: faces is with index 0) or it is the label name as string
    # both are added to the map, user can decide which to use
    # value is the list of ids belonging to that category
    labelled_images = defaultdict(list)
    dataset_named_categories = dataset.categories
    for i in range(len(dataset)):
        img, label = dataset[i]
        # label returns the array index for dataset.categories
        # labelled_images[str(label)].append(i)
        category_name = dataset_named_categories[label].lower()
        labelled_images[category_name].append(i)
    return (dataset, labelled_images)

def get_image_categories():
    dataset = torchvision.datasets.Caltech101(root='./data', download=True, target_type='category')
    labelled_images = defaultdict(list)
    dataset_named_categories = dataset.categories 
    for i in range(len(dataset)):
        _, label = dataset[i]
        category_name = dataset_named_categories[label]
        labelled_images[i] = category_name
    return labelled_images

def display_image_og(pil_img)->Image:
    pil_img.show()
    return pil_img

def find_nearest_square(k: int) -> int:
    return math.ceil(math.sqrt(k))

def gen_unique_number_from_title(string: str) -> int:
    a = 0
    for c in string:
        a+=ord(c)
    return a

def display_k_images_subplots(dataset: datasets.Caltech101, distances: tuple, title: str):
    pyplot.close()
    k = len(distances)
    # print(len(distances))
    # distances tuple 0 -> id, 1 -> distance
    split_x = find_nearest_square(k)
    split_y = math.ceil(k/split_x)
    # print(split_x, split_y)
    # this does not work
    # pyplot.figure(gen_unique_number_from_title(title))
    fig, axs = pyplot.subplots(split_x, split_y)
    fig.suptitle(title)
    ii = 0
    for i in range(split_x):
        for j in range(split_y):
            if(ii < k):
                # print(ii)
                id, distance = distances[ii][0], distances[ii][1]
                img, _ = dataset[id]
                if(img.mode == 'L'): axs[i,j].imshow(img, cmap = 'gray')
                else: axs[i,j].imshow(img)
                axs[i,j].set_title(f"Image Id: {id} Distance: {distance:.2f}")
                axs[i,j].axis('off')
                ii += 1
            else:
                fig.delaxes(axs[i,j])
    # pyplot.title(title)
    pyplot.show()

def get_user_selected_feature_model(): 
    """This is a helper code which prints all the available fearure options and takes input from the user"""
    print('Select your option:\
        \n\n\
        \n1. Color Moments\
        \n2. Histogram of Oriented gradients\
        \n3. RESNET-50 Layer3\
        \n4. RESNET-50 Avgpool\
        \n5. RESNET-50 FC\
        \n\n')
    option = int_input()
    model_space = None
    match option:
        case 1: model_space = torch.load('color_moments.pkl')
        case 2: model_space = torch.load('hog.pkl') 
        case 3: model_space = torch.load('layer3_vectors.pkl') 
        case 4: model_space = torch.load('fc_layer_vectors.pkl') 
        case 5: model_space = torch.load('fc_layer_vectors.pkl') 
        case default: print('No matching input was selected')
    return model_space, option

def get_user_input_k():
    """Helper function to be used in other functions to take input from user for K"""
    print("Please enter the value of K : ")
    value = int_input()
    return value

def get_user_selected_dim_reduction():
    """This is a helper code which prints all the available Dimension reduction options and takes input from the user"""
    print('Select your option:\
        \n\n\
        \n1. SVD - Singular Value Decomposition\
        \n2. NNMF - Non Negative Matrix Factorization\
        \n3. LDA - Latent Dirichlet Allocation\
        \n4. K - Means\
        \n\n')
    option = int_input()
    return option


def label_feature_descriptor_fc(): 

    '''
    Loads data from pickle file that contains RESNET FC feature vector along with label id and name for each image. 
    Format of the pickle file : {'id': 8676, 'label_id': 100, 'label': 'yin_yang', 'ResnetFC': tensor([])}
    Total 101 labels from 0 to 100
    This is to be replaced by data from DB
    '''
    
    #Loading data
    fc_with_labels = torch.load('fc_labels.pkl')

    #Stores info of every label in form of a dictionary and appends all the entries to a list 
    list_of_label_dictionaries = []
    
    #Gets all the unique label names and the number of labels from the data in ordered format
    labels = list(OrderedSet([entry['label'] for entry in fc_with_labels]))
    number_of_labels = len(labels)
    
    #For all the images of a label_id create dictionary containing label_id, label and label feature descriptor
    #Append the dictionary in to a list 
    for label_id in range(number_of_labels):
        
        label_dict = {}
        
        #Creates a list of all the feature vectors for a particular label id
        label_feature_descriptors = [ entry['ResnetFC'].flatten() for entry in fc_with_labels if entry['label_id'] == label_id ]
        
        #Stacks into one tensor and takes mean to ouput a label feature descriptor
        label_super_descriptor = torch.stack(label_feature_descriptors).mean(0)
        
        #Create the dictionary with details about the label and append to list
        label_dict['label_id'] = label_id 
        label_dict['label'] = labels[label_id]
        label_dict['label_feature'] = label_super_descriptor
        
        list_of_label_dictionaries.append(label_dict)
    
    
    #Saving to a pkl file 
    torch.save(list_of_label_dictionaries,'fc_labels_mean.pkl')
    