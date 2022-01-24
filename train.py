import os
import json
from socket import SOL_CAN_BASE
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34

import argparse
import moxing as mox 
import numpy as np
from deep_moxing.model_analysis.api import analyse, tmp_save


task_type_out = None
pred_list_out = None
label_list_out = None
name_list_out = None
label_dict_out = None
eval_url_out = None


def main(*arg):
    global task_type_out, pred_list_out, label_list_out, name_list_out, label_dict_out, eval_url_out
    # 创建解析
    parser = argparse.ArgumentParser(description="train mnist",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--train_url', type=str, 
                    default='obs://test22161/model',
                    help='the path model saved')
    parser.add_argument('--data_url', type=str, 
                    default='obs://test22161/data_set',
                    help='the training data')
    parser.add_argument('--eval_url', type=str, 
                    default='obs://test22161/out',
                    help='the training data')
    # 解析参数
    args, unkown = parser.parse_known_args()
    train_url = args.train_url
    data_url = args.data_url
    eval_url = args.eval_url
    eval_url_out = eval_url
    #下载云上数据参数至容器本地
    mox.file.copy_parallel('obs://test22161/data_set', data_url)

    # 评估代码需要tensorflow包
    #a = os.system(r'conda install cudatoolkit=10.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/')
    #print("a : "+str(a))
    a = os.system(r'pip install -U --ignore-installed wrapt enum34 simplejson netaddr')
    print("a : "+str(a))    # 返回值为0表示下载成功
    a = os.system(r'pip install tensorflow')
    print("a : "+str(a))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    image_path = data_url

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(data_url, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = os.path.join(data_url, "resnet34-pre.pth")
    
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = os.path.join(train_url, 'resNet34.pth')
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    # 由于对代码不熟悉 这里选择直接重新来一遍 不洗牌地预测，获得预测值
    train_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    
    pred_list = []
    label_list = []
    net.eval()
    with torch.no_grad():
        val_bar = tqdm(train_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            pred_list += predict_y.cpu().numpy().tolist()
            label_list += val_labels.to(device).cpu().numpy().tolist()


    pred_list_out = np.array(pred_list)     #1 # 转化成需要的格式
    label_list_out = np.array(label_list)   #2
    name_list_out = train_loader.dataset.samples    #3
    for idx in range(len(name_list_out)):
        name_list_out[idx] = name_list_out[idx][0]
    label_dict = {}                         #4
    for idx,item in enumerate(label_list):
        label_dict[idx] = int(item)
    label_dict_out = json.dumps(label_dict)

    #上传容器本地数据至云上obs路径
    mox.file.copy_parallel(train_url, 'obs://test22161/resnet/model')

    print('Finished Training')


def evalution():
    # 评估模型的代码，参数解析
    # task_type = 'image_classification'
    # pred_list 预测标签
    # label_list 实际标签
    # name_list  文件路径列表
    # label_map_dict json序列化后的label_dict
    task_type = 'image_classification'
    print(type(task_type))  
    print(type(pred_list_out))
    print(type(label_list_out))  
    print(type(name_list_out))   
    print(label_dict_out)  
    
    # analyse
    res = analyse(
        task_type=task_type,
        pred_list=pred_list_out,
        label_list=label_list_out,
        name_list=name_list_out,
        label_map_dict=label_dict_out,
        save_path=eval_url_out)

    # 将模型预测结果写道obs中
    mox.file.copy_parallel(eval_url_out, 'obs://test22161/out')


if __name__ == '__main__':
    main()
    evalution()
