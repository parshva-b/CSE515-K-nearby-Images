from typing import TypedDict
import heapq

import torch
import numpy as np
from k_mediods import kmedoids
from tqdm import tqdm
from Mongo.mongo_query_np import get_all_feature_descriptor_for_label

from distances import cosine_similarity
from utils import (
    get_user_selected_feature_model,
    get_user_selected_feature_model_only_resnet50_output,
    convert_higher_dims_to_2d,
    feature_model,
)


def label_fv_kmediods(label: str, dbName: str):
    """
    create label feature vector using the kmediods value of the contained data
    return that feature vector
    """
    try:
        label_features = get_all_feature_descriptor_for_label(dbName, label)
        label_features = convert_higher_dims_to_2d(label_features)
        kmedoids_object = kmedoids()
        kmedoids_centroids = kmedoids_object.fit(label_features)
        # kmedoids_centroids = KMedoids(n_clusters=1).fit(label_features).cluster_centers_
        # print(kmedoids.shape, type(kmedoids))
        return kmedoids_centroids
    except KeyError:
        return None


def label_image_distance_using_cosine(
    max_len: int, label_feature_vectors, dict_all_feature_vectors, k: int
):
    distances = []
    for i in tqdm(range(max_len)):
        distances.append(
            cosine_similarity(
                label_feature_vectors.flatten(),
                np.asarray(dict_all_feature_vectors[i]).flatten(),
            )
        )
    top_k = heapq.nlargest(k, enumerate(distances), key=lambda x: x[1])
    # print(top_k)
    return top_k


def create_labelled_feature_vectors(labelled_images, task2bSpecific: bool = False):
    """
    labelled_images from map -> label_id : [images_ids]
    output format: [dict[key: label, value: feature_vector_label], model_space]
    """
    model_space = None
    if task2bSpecific:
        (
            model_space,
            option,
            dbName,
        ) = get_user_selected_feature_model_only_resnet50_output()
    else:
        model_space, option, dbName = get_user_selected_feature_model()
    # print(f'Model space shape {model_space.shape}')
    if model_space is not None:
        labelled_feature_vectors = {}
        for key in labelled_images.keys():
            # combine all feature vectors into one for label -- labelled embedding
            # add your min, mean, max code for label here
            labelled_feature_vectors[key] = label_fv_kmediods(key, dbName)
        return (labelled_feature_vectors, model_space, option)
    else:
        return None


def get_all_label_feature_vectors(labelled_images):
    """Gets all the feature vectors output shold be 101 x 1000 Fc_layer
    input : labelled_images
    output : 101 x 1000 Ndarray"""
    labelled_feature_vectors = []
    for key in labelled_images.keys():
        # combine all feature vectors into one for label -- labelled embedding
        # add your min, mean, max code for label here
        labelled_feature_vectors.append(label_fv_kmediods(key, feature_model[5])[0])
    return labelled_feature_vectors
