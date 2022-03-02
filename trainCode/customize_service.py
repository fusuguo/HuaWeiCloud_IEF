import os
import json

import torch
from PIL import Image
from torchvision import transforms

import log
from model_service.pytorch_model_service import PTServingBaseService
import torch.nn.functional as F
import torch.nn as nn
from model import resnet34
logger = log.getLogger(__name__)


data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTVisionService, self).__init__(model_name, model_path)
        self.model = MyResnet(model_path)
        self.label = [0,1,2,3,4,5,6,7,8,9]
        dir_path = os.path.dirname(os.path.realpath(self.model_path))
        print('dis_path: '+dir_path)
        with open(os.path.join(dir_path, 'class_indices.json')) as f:
            self.label = json.load(f)


    def _preprocess(self, data):
        
        preprocessed_data = {}
        for k, v in data.items():
            input_batch = []
            for file_name, file_content in v.items():
                with Image.open(file_content) as image1:
                    image1 = data_transform(image1)
                    #image1 = torch.unsqueeze(image1, dim=0)
                    #image1 = torch.squeeze(image1, dim=0)
                    input_batch.append(image1)
            input_batch_var = torch.stack(input_batch, dim=0)

            print(input_batch_var.shape)
            preprocessed_data[k] = input_batch_var

        return preprocessed_data

    def _postprocess(self, data):
        results = []
        for k, v in data.items():
            print(v)
            result = torch.argmax(v)
            print(result.shape)
            print(self.label)
            result = {k: self.label[ str(int(result.numpy().reshape(1,-1)[0])) ]}
            results.append(result)
        return results

    def _inference(self, data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        result = {}
        for k, v in data.items():
            # prediction
            with torch.no_grad():
                # predict class
                output = torch.squeeze(self.model(v.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                print("predict:")
                print(predict)
                result[k] = predict

        return result



def MyResnet(model_path, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet34(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
    model.eval()
    return model
    
