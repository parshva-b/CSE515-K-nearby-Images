import utils
import torch
import label_vectors
from Mongo import mongo_query_np
from distances import cosine_distance


class Task10:
    def __init__(self) -> None:
        self.dataset, self.labelled_images = utils.initialise_project()

    def compute_closet_distance(self, cur_label_vector, all_vectors, k) -> list:
        distances = []
        for i in range(len(all_vectors)):
            cur_distance = cosine_distance(
                cur_label_vector.flatten(), all_vectors[i].flatten()
            )
            distances.append((cur_distance, i))
        distances.sort()

        top_k = []
        for i in range(k):
            top_k.append((distances[i][1], distances[i][0]))
        return top_k

    def get_closet_image(self, cur_label_vector):
        all_image_vectors = mongo_query_np.get_all_feature_descriptor(
            utils.feature_model[5]
        )
        closest = self.compute_closet_distance(cur_label_vector, all_image_vectors, 1)
        print(f"Most similar image with distance: {closest}")
        return closest[0][0]

    def runTask10(self) -> None:
        # take label input
        # take semantic label input
        # take user input of k
        # display top k image

        # for key in self.labelled_images: print(key)
        label_index_selected = utils.get_user_input_label()
        label_selected = self.dataset.categories[label_index_selected]
        print(f"Input provided: {label_index_selected} => {label_selected}")
        if label_selected not in self.labelled_images:
            print("label not in database -> try again")
            return

        pathname, option = utils.get_user_input_latent_semantics()
        if pathname is None:
            return

        # create and load label vector for current label in question
        # always selected fc layer (gives the best result)
        cur_label_fv = label_vectors.label_fv_kmediods(
            label_selected, utils.feature_model[5]
        )

        k = utils.get_user_input_k()
        latent_model_space = torch.load(pathname)

        print(f"Found label feature in fc layer with shape: {cur_label_fv.shape}")

        top_k_distances = []

        match option:
            case 1:
                # get closest image for each option
                closest_image_id = self.get_closet_image(cur_label_fv)
                print(
                    f"label range: {self.labelled_images[label_selected][0]} - {self.labelled_images[label_selected][-1]}"
                )
                # LS1 - compare cur_label_fv with latent_model_space
                # use latent_model_space
                # get top k images
                print(closest_image_id)
                closest_image_vector = latent_model_space[closest_image_id // 2]
                top_k_distances = self.compute_closet_distance(
                    closest_image_vector, latent_model_space, k
                )
            case 2:
                # get closest image for each option
                closest_image_id = self.get_closet_image(cur_label_fv)
                print(
                    f"label range: {self.labelled_images[label_selected][0]} - {self.labelled_images[label_selected][-1]}"
                )
                print(closest_image_id)
                latent_model_space = latent_model_space[1][0]
                closest_image_vector = latent_model_space[closest_image_vector // 2]
                top_k_distances = self.compute_closet_distance(
                    closest_image_vector, latent_model_space, k
                )
            case 4:
                # get closest image for each option
                closest_image_id = self.get_closet_image(cur_label_fv)
                print(
                    f"label range: {self.labelled_images[label_selected][0]} - {self.labelled_images[label_selected][-1]}"
                )
                print(closest_image_id)
                closest_image_vector = latent_model_space[closest_image_id // 2]
                top_k_distances = self.compute_closet_distance(
                    closest_image_vector, latent_model_space, k
                )
            case 3:
                # get closest label for latent space
                # loop over entire db and get top k images
                closest_label_index_for_selected_label = (
                    self.compute_closet_distance(
                        latent_model_space[label_index_selected], latent_model_space, 2
                    )
                )[1][0]
                closest_label_for_selected_label = self.dataset.categories[
                    closest_label_index_for_selected_label
                ]
                print(
                    f"Considering closest label {closest_label_index_for_selected_label} -- {closest_label_for_selected_label}"
                )
                print(
                    f"label range: {self.labelled_images[closest_label_for_selected_label][0]} - {self.labelled_images[closest_label_for_selected_label][-1]}"
                )

                collection_name_in_consideration = utils.feature_model[5]
                cur_ls_model = label_vectors.label_fv_kmediods(
                    closest_label_for_selected_label, collection_name_in_consideration
                )

                all_features = mongo_query_np.get_all_feature_descriptor(
                    collection_name_in_consideration
                )
                top_k_distances = self.compute_closet_distance(
                    cur_ls_model, all_features, k
                )
            case default:
                print("yeh toh bada toi hai!!")

        top_k_distances = [(id * 2, distance) for id, distance in top_k_distances]
        print(top_k_distances)
        utils.display_k_images_subplots(
            self.dataset, top_k_distances, "Top k images for label"
        )


if __name__ == "__main__":
    task10 = Task10()
    task10.runTask10()