import torch
import torchvision.models as models 
import torchvision.transforms as transforms
from utils import *
import numpy as np
from torchvision import datasets
from tqdm import tqdm


class resnet_features():
    def __init__(self) -> None:
        self.hook_layer3_output = []
        self.hook_avgpool_output = []
        self.hook_fc_layer_output = []
        pass

    def get_tensor_image(self, image):
        """Gets images and converts them into tensor image"""
        image = check_rgb_change_grayscale_to_rgb(image)
        cv2_image = np.array(image)
        # Creating squence of image transformation to convert image to tensor and resize it
        image_transformer = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
        tensor_image = image_transformer(cv2_image)
        # adding one more dimenestion to the image input as the model takes only batch input
        model_input_image = tensor_image.unsqueeze(0)
        return model_input_image


    def run_model (self, image):
        """This function runs hook with all three hooks so that we don't have to run it every time when we pull the feature vectors"""
        
        def layer3_hook(module, input, output):
            self.hook_layer3_output.append(output)
        def avgpool_hook(module, input, output):
            self.hook_avgpool_output.append(output)
        def fc_layer_hook(module, input, output):
            self.hook_fc_layer_output.append(output)
        # Getting the image
        model_input_image = self.get_tensor_image(image)

        resnet50_model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        resnet50_model.eval()

        avgpool_layer = resnet50_model.avgpool
        avgpool_layer.register_forward_hook(avgpool_hook)

        resnet_layer3 = resnet50_model.layer3
        resnet_layer3.register_forward_hook(layer3_hook)

        resnet_fc_layer = resnet50_model.fc
        resnet_fc_layer.register_forward_hook(fc_layer_hook)

        with torch.no_grad():
            val = resnet50_model(model_input_image)
            

    def resnet_avgpool(self):
        """Get feature vectors from avgpool layer of the """

        vector_output = self.hook_avgpool_output[0].squeeze()
        vector_1024 = torch.mean(vector_output.view(1024, -1, 2), dim=2)
        
        vector_1024 = vector_1024.reshape(-1)
        vector_1024 = np.array(vector_1024)

        # print(vector_1024)
        # print(type(vector_1024))

        return vector_1024


    def resnet_layer3(self):
        """Gets feature vectors from layer3 layer of resnet"""

        vector_output = self.hook_layer3_output[0].squeeze()
        vector_1024 = torch.mean(vector_output, dim=(1,2))
        vector_1024 = np.array(vector_1024)

        # print(vector_1024.shape)
        # print(type(vector_1024))

        return vector_1024


    def resnet_fc_layer(self):
        """Get feature vectors from fc layer""" 

        vector_output = self.hook_fc_layer_output[0].squeeze()
        vector_1000 = np.array(vector_output)


        return vector_1000 
    
    def apply_softmax(self):
        """
        Returns ndarray with shape 1000
        Final output of resent50 model - Applying Softmax on FC layer.
        """
        vector_output = self.hook_fc_layer_output[0].squeeze()
        vector_1000 = np.array(vector_output)
        softmax_values = self.softmax(vector_1000)
        return softmax_values

    def softmax(self, fc_layer):
        exp_vector = np.exp(fc_layer)
        softmax_values = exp_vector / np.sum(exp_vector)
        return softmax_values


if __name__ == "__main__":
    temp  = resnet_features()
    pickle_file_path = "fc_layer_vectors.pkl"
        # opening the pickle files for fc layer
    with open(pickle_file_path, 'rb') as file:
        feature_data = torch.load(file)
    
    softmax_dict = {}
    for imageid in tqdm(range(8677)):
        softmax_value = temp.softmax(np.array(feature_data[imageid]))
        softmax_dict[imageid] = softmax_value
        print(imageid)
    # print(softmax_dict)
    resnet_pickle_path = "resnet_vectors.pkl"
    torch.save(softmax_dict, resnet_pickle_path)
    print("Features are saved")
