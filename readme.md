# 解决方案介绍
## 1. 算法设计
项目首先通过PolarMask得出视频第一帧的annotation，然后通过对frtm-vos进行改进，设计mtmfi-vos网络对视频进行半监督目标分割，最终得到视频所有帧的annotation。
## 2. 模型结构
基于frtm-vos改进的mtmfi-vos网络结构如下：

![Alt text](./user_data/tmp_data/mtmfi-vos.jpg)

主要创新点如下：
1. 设计多尺度粗分割模型 (Multi-scale Target Models, MTM) 捕获更多目标外观细节，以下成MTM
2. 设计特征整合模块 (feature integration, FI) 突出帧间目标动态变化情况，以下成FI

## 3. 训练推理
### 1） 训练阶段
由于时间紧迫（4月10日才参加），仅使用Davis数据与Youtube数据集进行训练，epoch为260，优化器为Adam，损失函数为BCEloss，学习率为1e-3（前127个epoch，而后衰减至1e-4）

项目共提供3个训练模型：
* resnet101_all_bise_mini_ep0205.pth：使用YouTube+Davis进行训练，网络包含MTM和FI结构
* resnet101_all_l3+l4_ep0260.pth：使用YouTube+Davis进行训练，网络仅包含MTM结构
* resnet101_dv_l3+l4_ep0260.pth：仅使用Davis进行训练，网络包含MTM和FI结构

### 2） 推理阶段
#### 1. 将test的videos和各个videos的第一帧annotation（由PolarMask生成）缩放至480p（文件路径在code/resize.py内修改）
    python code/resize.py
#### 2. 进行测试集的推理
    python code/inference.py --dev cuda:0 --dset ali2021val --seg bise --model resnet101_all_bise_mini_ep0205.pth 
注：  
+ --dev：GPU配置  
+ --dset：数据集选择：   
本项目-***ali2021val***  
Davis16-***dv2016val***  
Davis17-***dv2017val***  
Youtube-***yt2018val***  
+ --seg：分割模块选择：  
包含MTM模块-***initial***   
包含MTM以及FI模块-***bise***  
+ --model：模型选择  
resnet101_all_bise_mini_ep0205.pth  
resnet101_all_l3+l4_ep0260.pth  
resnet101_dv_l3+l4_ep0260.pth  
#### 3. 将推理结果恢复至原分辨率（文件路径在code/inverse_resize.py内修改）
    python code/inverse_resize.py
## 4. 实现细节
团队软硬件配置：  
GPU：Nvidia RTX2080Ti CUDA10.0+cudnn7.6.4  
服务器环境：Ubuntu 18.04.5 LTS 
PyTorch版本：1.6.0  
必要安装库：  
scipy==1.5.2   
scikit-image==0.17.2   
tqdm==4.60.0   
opencv-python==4.4.0.42   
easydict==1.9   
numpy==1.19.1  

**测试集问题**：  
”639110“序列缺少第二帧 复制第三帧结果至第二帧