import os
import json

import torch
from PIL import Image
from torchvision import transforms
# import matplotlib.pyplot as plt
# from collections import defaultdict, OrderedDict
import json
from model import resnet34
import re


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")       

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = "./pic"
    # imgs_root = r"C:\Users\95340\Downloads\oxford-102-flowers\jpg"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg") or i.endswith(".jpg")]

    # print(len(img_path_list))

    aa = 0
    for img_path in img_path_list:

        # video = defaultdict(list)
       
        data_transform = transforms.Compose(    
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        aa +=1
        # print('\n')
        newName_jpg = re.findall(r'[^\\/:*?"<>|\r\n]+$',img_path)
        # print(newName_jpg)
        newName =  re.findall(r'(.+?)\.jpg',newName_jpg[0])
        # print(newName)
        vi = {"imageName":"".join(newName_jpg)}

        # [N, C, H, W]
        img = Image.open(img_path)
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)
        
        # create model
        model = resnet34(num_classes=5).to(device)

        # load model weights
        weights_path = "./resNet34.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        # prediction
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            probs, classes = torch.max(predict, dim=0)
      
        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                             predict[predict_cla].numpy())

        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                             predict[i].numpy()))
        # video["image"].append(''.join(newName_jpg))
        # video["class"].append(class_indict[str(predict_cla)])
        # video["score"].append(probs.numpy()+0)

        vi["class"] = class_indict[str(predict_cla)]
        vi["prob"] = str(probs.numpy()+0)

        new_floder = "./result"
        if os.path.exists(new_floder):
            pass
        else:
            os.mkdir(new_floder)
        
        json_name = new_floder + "//" + ''.join(newName) + ".json"

        # test_dict = video
        # json_str = json.dumps(test_dict, indent=4)

        with open(json_name, 'w') as json_file:
            json_file.write(json.dumps(vi,indent=4))
   
    # test_dict = {
    # "name":"ResNet_classification",
    # 'version': "1.0",
    # 'results': video
    # }   
    
    print("Done")

    


if __name__ == '__main__':
    main()
