# 编辑记录

将各个阶段的编辑记录按照工作流程的顺序记录下来

<a href="#准备数据">准备数据</a>

<a href="#适配算法代码">适配算法代码</a>

<a href="#创建算法">创建算法</a>

<a href="#训练模型">训练模型</a>

<a href="#部署AI应用">部署AI应用</a>

<a href="#预测结果">预测结果</a>



## <a id="准备数据">准备数据</a>

创建`obs`桶对象，建立四个文件夹。目录如下

```
OBS桶/目录名
|── resnet 存放模型输出
|   ├── model  必选： 固定子目录名称，用于放置模型相关文件
|   │  ├── <<自定义Python包>>  可选：用户自有的Python包，在模型推理代码中可以直接引用
|   │  ├── resNet34.pth 必选: 模型文件夹，包含pyspark保存的模型内容
|   │  ├── config.json 必选：模型配置文件，文件名称固定为config.json, 只允许放置一个
|   │  ├── customize_service.py  可选：模型推理代码，文件名称固定为customize_service.py, 只允许放置一个，customize_service.py依赖的文件可以直接放model目录下
|	|  |—— model.py 必选
|	|  |—— class_indices.py 必选

|—— code 存放训练文件
|	|—— train.py 必选
|	|—— model.py 必选


|—— data_set 存放数据集
|	|—— resnet34-pre.pth 必选
|	|—— train 训练集
|	|	|—— daisy
|	|	|—— dandelion
|	|	|—— roses
|	|	|—— sunflowers
|	|	|—— tulips
|	|—— val	验证集
|	|	|—— daisy
|	|	|—— dandelion
|	|	|—— roses
|	|	|—— sunflowers
|	|	|—— tulips

|—— out 存放评估文件
```

## <a id="适配算法代码">适配算法代码</a>

主要针对`train.py`进行了适配。添加的代码不是最优，有可以简化的地方。

line20~25，设置几个全局参数，为评估所需的函数做准备

```python
task_type_out = None
pred_list_out = None
label_list_out = None
name_list_out = None
label_dict_out = None
eval_url_out = None
```

line28，为了解析参数，添加`*args`。

line29，声明全局变量，使得`main`函数内可以修改这几个全局变量的值。

line30~38，解析参数，一共有三个参数，*这三个参数名和后面算法创建和训练作业创建的参数名相同*。三个参数的含义如下

- `data_url`：容器中训练集的位置
- `train_url`：容器中保存模型的位置
- `eval_url`：容器中保存评估文件的位置

可以自行添加想要的参数，示例如下

```python
parser.add_argument('--data_url', type=str, 
                default='obs://resnet-576f/data_set',
                help='the training data')
```

line49\~50，line207\~208，line236\~237，利用`moxing`接口，将云上数据下载到容器或者将容器数据上传到云端。如下

```python
mox.file.copy_parallel(source_url, destination_url)
```

line52~58，由于添加的评估代码需要`tensorflow`包，在代码中利用pip指令下载，返回值为0表示下载成功。

line77，将`image_path`改为`data_url`。line113和line134，修改了文件路径。这三处修改了文件路径。

 line179~line205，给评估函数准备数据。首先**在shuffle为false**的情况下，给train文件夹的数据预测标签。这里选择了直接复制已有代码的方式，如下

```python
 # 由于对代码不熟悉 这里选择直接重新来一遍 不洗牌地预测，获得预测值
 train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
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
```

然后对得到的数据进行一定的处理

```python
pred_list_out = np.array(pred_list)     #1 # 转化成需要的格式
label_list_out = np.array(label_list)   #2
name_list_out = train_loader.dataset.samples    #3
for idx in range(len(name_list_out)):
    name_list_out[idx] = name_list_out[idx][0]
label_dict = {}                         #4
for idx,item in enumerate(label_list):
    label_dict[idx] = int(item)
label_dict_out = json.dumps(label_dict)
```

line213~237，添加评估函数。主要是调用了`analyse()`函数。

line242，调用`evalution()`函数。





## <a id="创建算法">创建算法</a>

需要手动配置的地方如下

![](./pics/1.png)





## <a id="训练模型">训练模型</a>

选择上面配置的算法后，需要手动配置的地方如下

![](./pics/2.png)



#### 关于模型评估

关于模型的评估，评估结果包括三个文件

- `model_analysis_results.json`：主要的评估文件
- `all_samples.json`：存放文件路径名及其对应标签
- `tmp_model_analysis.npz`

> 评估结果在训练作业的在线渲染尝试失败
>
> 由于`model_analysis_results.json`文件太大，可以选择一些json可视化工具打开

提供两个评估文件夹，分别是用`train`和`val`进行评估。用`val`进行评估的结果文件较小，便于查看。

```
|—— analysis-train
|	|—— model_analysis_result.json
|	|—— all_samples.json
|	|—— tmp_model_analysis.npz
|—— analysis-val
|	|—— model_analysis_result.json
|	|—— all_samples.json
|	|—— tmp_model_analysis.npz
```

在训练作业详情界面可以看到运行时间（包括了代码中上传数据、下载数据、下载依赖包的时间）

训练作业界面可以看到资源利用率，如下

![](./pics/3.png)





## <a id="部署AI应用">部署AI应用</a>



推理代码已经自动选择（推理代码的目录需要设置为`./resnet/model/customize_service.py`）。其他需要设置的内容为

1. 训练作业：选择对应的训练作业
2. AI引擎：设置为`PyTorch+python3.6`



### 关于推理代码

将`predict.py`改写为`customize_service.py`

> 参考
>
> [模型推理代码编写说明](https://support.huaweicloud.com/inference-modelarts/inference-modelarts-0057.html#inference-modelarts-0057)
>
> [`pytorch`自定义代码编写实例](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0175.html)

line9，`PyTorch`下需要继承`PTServingBaseService`类。

```python
from model_service.pytorch_model_service import PTServingBaseService
```

几个方法介绍如下。

`__init__()`，主要建立模型、获得标签字典。

`_preprocess()`，处理原始数据。

`_postprocess()`，在line60把result处理为`self.label`所需要的key形式

`_inference()`，主要的推断函数

`MyResnet()`，获得模型



## <a id="预测结果">预测结果</a>

发布在线模型不需要其他的设置。

待在线任务发布成功后，预测结果如下

![](./pics/4.png)

[关于接口调用，可参考](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0063.html#modelarts_23_0063__fig32966269191)

















