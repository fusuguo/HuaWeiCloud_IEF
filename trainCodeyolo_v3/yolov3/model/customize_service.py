from PIL import Image
import log
from model_service.pytorch_model_service import PTServingBaseService
import torch.nn.functional as F

import torch.nn as nn
import torch
import json

import numpy as np

import models
import cv2
from build_utils import img_utils, torch_utils, utils


logger = log.getLogger(__name__)

import torchvision.transforms as transforms

# import importlib

# importlib.reload(sys)


img_size = 512  # 
cfg = "cfg/my_yolov3.cfg"  # 
weights = "weights/yolov3spp-voc-512.pt"  # 
json_path = "data/pascal_voc_classes.json"  # 
img_path = "test.jpg"
input_size = (img_size, img_size)


import os


class PTVisionService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTVisionService, self).__init__(model_name, model_path)
        global cfg,weights,json_path
        dir_path = os.path.dirname(os.path.realpath(self.model_path))
        cfg = os.path.join(dir_path,cfg)
        weights = os.path.join(dir_path,cfg)
        json_path = os.path.join(dir_path,json_path)

        self.model = Mnist(model_path)
        json_file = open(json_path, 'r')
        class_dict = json.load(json_file)
        json_file.close()
        self.label = {v: k for k, v in class_dict.items()}
        self.imgshape = {}
        # number:object


    def _preprocess(self, data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        preprocessed_data = {}
        for k, v in data.items():
            img = torch.zeros((1, 3, img_size, img_size), device=device)
            self.model(img)

            img_o = cv2.imread(img_path)  # BGR
        #5  
            name = list(v.keys())
            content = list(v.values())
            testpath = os.path.realpath(name[0])

            with open(testpath,mode="wb+") as file:
                file.write(content[0].getvalue())

            img_o = cv2.imread(testpath)  # BGR
            assert img_o is not None, "Image Not Found " + img_path

            img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).float()
            img /= 255.0  # scale (0, 255) to (0, 1)
            img = img.unsqueeze(0)  # add batch dimension
            self.imgshape[k] = [img,img_o]

            preprocessed_data[k] = img
        return preprocessed_data

    def _postprocess(self, data):
        # results = []
        for k, v in data.items():
            pred = utils.non_max_suppression(v, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
            if pred is None:
                print("No target detected.")
                exit(0)

            img = self.imgshape[k][0]
            img_o = self.imgshape[k][1]
            pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()

            bboxes = pred[:, :4].detach().cpu().numpy()
            scores = pred[:, 4].detach().cpu().numpy()
            classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1
            print(classes)
            box = {}
            box["detection_classes"] = [self.label[x] for x in classes]
            box["detection_boxes"] = bboxes.tolist()
            box["detection_scores"] = scores.tolist()
            results = box
            # results.append(result)
        return results

    def _inference(self, data):

        result = {}
        for k, v in data.items():
            result[k] = self.model(v)[0]

        return result



def Mnist(model_path, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.Darknet(cfg, img_size)
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    model.to(device)

    model.eval()

    return model

