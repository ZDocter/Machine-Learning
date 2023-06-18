"""
1、Dataset的使用
2、tensorboard的使用
3、transforms的使用
4、DataLoader的使用
5、MaxPool层的使用
6、Conv2d的使用
7、ReLU层的使用
8、torchvision的使用
"""
# import os
# import time
# import numpy
# import torch
# import torchvision
# from torch import nn
# from torch.nn import MaxPool2d, ReLU, Sequential, Flatten, L1Loss, MSELoss, CrossEntropyLoss
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter
# import torch.nn.functional as F
# from PIL import Image
# import matplotlib.pyplot as plt
#
# from torchvision import transforms

"""---------------------Dataset--------------------"""
# # 所有子类必须重写__getitem__()和__len__()方法
# class MyData(Dataset):
#
#     """初始化"""
#     def __init__(self, root_dir, label_dir):
#         self.root_dir = root_dir
#         self.label_dir = label_dir
#         self.path = os.path.join(self.root_dir, self.label_dir)
#         self.path_list = os.listdir(self.path)  # 地址列表
#
#     """获取数据集中每一个"""
#     def __getitem__(self, idx):
#         name = self.path_list[idx]
#         item_path = os.path.join(self.root_dir, self.label_dir, name)
#         image = Image.open(item_path)
#         label = self.label_dir
#         return image, label
#
#     """返回数据地址列表长度"""
#     def __len__(self):
#         return len(self.path_list)
#
#
# # Ants dataset
# ants_dataset = MyData(root_dir=r'D:\PyTorch_Python\Dataset_packing\hymenoptera_data\train',
#                       label_dir=r'ants')
# # Bees dataset
# bees_dataset = MyData(root_dir=r'D:\PyTorch_Python\Dataset_packing\hymenoptera_data\train',
#                       label_dir=r'bees')
# # All dataset
# train_dataset = ants_dataset + bees_dataset
# image_ant0, label_ant0 = ants_dataset.__getitem__(0)  # ants_dataset[0]
# image_bee0, label_bee0 = bees_dataset.__getitem__(0)  # bees_dataset[0]
# print("The label is:", label_ant0, label_bee0)
# image_ant0.show()
# image_bee0.show()

"""---------------------tensorboard--------------------"""
# 使用tensorboard --logdir=log_dir激活视图
# 显示图像
# img = Image.open(r'D:\PyTorch_Python\Dataset_packing\hymenoptera_data\train\ants\0013035.jpg')
# img_array = numpy.array(img)
# print(img_array.shape)
# writer = SummaryWriter('logs')
# # writer.add_image('test', img_array, 1, dataformats='HWC')
#
# # 绘制函数图
# for i in range(100):
#     writer.add_scalar('y = 2x', i * 2, i)  # Name, y axis, x axis
# writer.close()

"""---------------------transforms--------------------"""
# img = Image.open(r'D:\PyTorch_Python\Dataset_packing\hymenoptera_data\train\ants\0013035.jpg')
# print(type(img))
# writer = SummaryWriter('logs_transforms')
#
# # 以下都是transforms模块中的类，所以在使用前先要进行实例化
# # ToTensor 将PILimage转为Tensor
# img_totensor = transforms.ToTensor()
# tensor_img = img_totensor(img)
# print(type(tensor_img))
# writer.add_image('原始图像', tensor_img, 0)
#
# # Normalize 正则化 参数：均值、标准差
# trans_norm = transforms.Normalize([0.5, 0.5, 0.5],
#                                   [0.5, 0.5, 0.5])
# normalized_img = trans_norm(tensor_img)
# print(type(normalized_img))
# writer.add_image('正则化图像', normalized_img, 0)
#
# # # Resize
# img_resize = transforms.Resize([512, 512])  # 变为512*512
# resized_img = img_resize(tensor_img)
# print(resized_img.shape)
# writer.add_image('resize图像', resized_img, 0)
#
# # Compose 将多个transform组合起来
# trans_resize_2 = transforms.Resize(512)  # 最小边变为512
# trans_compose = transforms.Compose([trans_resize_2,
#                                     transforms.ToTensor()])  # 将重构大小与转化为Tensor组合使用
# comp_trans_img = trans_compose(img)
# writer.add_image('重构大小', comp_trans_img, 1)
#
# writer.close()

"""---------------------torchvision中的数据集--------------------"""
# writer = SummaryWriter('logs_torchvision')
# # Train set
# train_set = torchvision.datasets.CIFAR10(r"D:\PyTorch_Python\BCI_DataSets",
#                                          transform=transforms.ToTensor(), train=True, download=True)
# # Test set
# test_set = torchvision.datasets.CIFAR10(r"D:\PyTorch_Python\BCI_DataSets",
#                                         transform=transforms.ToTensor(), train=False, download=True)
#
# print(type(test_set[0]))
# print(test_set.classes)  # 输出测试集的标签（keys）
# # Show the first 10 images of test_set
# # 该数据集中每个样本包含图片及标签
# for i in range(10):
#     img, target = test_set[i]
#     writer.add_image('CIFAR-10', img, i)
# writer.close()

"""---------------------DataLoader--------------------"""
# writer = SummaryWriter("logs_DataLoader")
# test_data = torchvision.datasets.CIFAR10(r"D:\PyTorch_Python\BCI_DataSets",
#                                          transform=torchvision.transforms.ToTensor(),
#                                          train=False, download=True)
#
# # Load dataset every 64 is a packing
# test_loader = DataLoader(dataset=test_data, batch_size=64,
#                          shuffle=True, num_workers=0, drop_last=False)
#
# # Extract data from test_loader
# step = 0
# for data in test_loader:
#     imgs, targets = data
#     # print(imgs.shape, targets)0
#     if step == 0:
#         print(imgs.shape)
#     writer.add_images("test_data", imgs, step)
#     step += 1
#
# writer.close()

"""---------------------卷积层 convolution layer--------------------"""
# """神经网络基本架构"""
# class My_Model(nn.Module):
#     # 构建模型必须重写forward()函数,定义每次执行的计算步骤
#     def __init__(self) -> None:
#         super().__init__()
#
#     # def __init__(self):
#     #     super(Model, self).__init__()
#
#     # Rewrite forward function
#     def forward(self, Input):
#         Output = Input + 1
#         return Output
#
# MyModel = My_Model()
# Output = MyModel(torch.tensor(1.0))
# print(Output)

# """卷积运算"""
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]])
#
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])
#
# print(input.shape, kernel.shape)
#
# input_1 = torch.reshape(input=input, shape=[1, 1, 5, 5])  # To 4D
# kernel_1 = torch.reshape(input=kernel, shape=[1, 1, 3, 3])  # To 4D
# print(input_1, input_1.shape, '\n', kernel_1, kernel_1.shape)
#
# output = F.conv2d(input=input_1, weight=kernel_1)
# output_1 = F.conv2d(input=input_1, weight=kernel_1, stride=2)
# output_2 = F.conv2d(input=input_1, weight=kernel_1, padding=1)
#
# print(output, '\n', output_1, '\n', output_2)

# """卷积网络构建"""
# writer = SummaryWriter('convolution layer')
#
# # Download datasets
# train_set = torchvision.datasets.CIFAR10(r"D:\PyTorch_Python\BCI_DataSets", train=True,
#                                          transform=transforms.ToTensor(), download=True)
# print('train_set:', train_set)
#
# # Load dataset
# train_load = DataLoader(dataset=train_set, batch_size=64,
#                         shuffle=True, num_workers=0, drop_last=False)
# print('type of train_load:', type(train_load))


# # Create CNN model
# class CNN_Model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         # 卷积核的数值根据输入数据及设定尺寸由输入数据中迭代产生
#         # Define a convolution layer
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
#
#     def forward(self, Input):
#         # Forward use the conv1 layer
#         Output = self.conv1(Input)
#         return Output
#
#
# Conv2d_model = CNN_Model()
# print(Conv2d_model)
#
# # Show images in tensorboard and train images
# step = 0
# for train_img, train_target in train_load:
#     output = Conv2d_model(train_img)
#     # print(output.shape)
#     writer.add_images('train_set', train_img, step)
#     output_reshape = torch.reshape(output, [-1, 3, 30, 30])
#     writer.add_images('Conv2d', output_reshape, step)
#     if step == 0:
#         print(train_img.shape)
#     step += 1
#
# # Train network model
# writer.close()

"""---------------------最大池化层 MaxPooling layer--------------------"""
# Input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)  # RuntimeError: "max_pool2d" not implemented for 'Long'
#
# Input = torch.reshape(Input, [-1, 1, 5, 5])
# print(Input.shape)
#
#
# # Network
# class MaxPool_Model(nn.Module):
#     def __init__(self) -> None:
#         super(MaxPool_Model, self).__init__()
#         self.Maxpool = MaxPool2d(kernel_size=3, ceil_mode=True)
#
#     def forward(self, Input):
#         output = self.Maxpool(Input)
#         return output
#
#
# My_Model = MaxPool_Model()
# Output = My_Model(Input)
# print(Output, Output.shape)

"""---------------------非线性层 Nonlinear layer--------------------"""
#
# Input = torch.tensor([[1, -0.5],
#                       [-1, 1]])
# Input = torch.reshape(Input, [-1, 1, 2, 2])
# print(Input.shape)
#
#
# class ReLU_Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.relu = ReLU()  # 将小于1的全部转为0
#
#     def forward(self, Input):
#         output = self.relu(Input)
#         return output
#
#
# model = ReLU_Model().forward(Input)
# print(model, '\nn', Input)

"""---------------------简单CNN网络搭建--------------------"""
# datasets = torchvision.datasets.CIFAR10("D:\\PyTorch_Python\\BCI_DataSets", train=True,
#                                         transform=transforms.ToTensor(), download=True)
#
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
#         # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
#         # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
#         # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
#         # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#         # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
#         # self.relu = nn.ReLU()
#         # self.linear1 = nn.Linear(in_features=1024, out_features=64)
#         # self.linear2 = nn.Linear(in_features=64, out_features=10)
#         # self.flatten = Flatten()
#
#         self.model = Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             Flatten(),
#             nn.Linear(in_features=1024, out_features=64),
#             nn.Linear(in_features=64, out_features=10)
#         )
#
#     def forward(self, train):
#         # train = self.conv1(train)
#         # train = self.maxpool1(train)
#         # train = self.conv2(train)
#         # train = self.maxpool2(train)
#         # train = self.conv3(train)
#         # train = self.maxpool3(train)
#         # train = self.flatten(train)
#         # train = self.linear1(train)
#         # train = self.linear2(train)
#         train = self.model(train)
#         return train
#
#
# mymodel = Model()
# output = mymodel(torch.ones(64, 3, 32, 32))
# print(output.shape)
#
# write = SummaryWriter('logs_CNN')
#
# write.add_graph(model=mymodel, input_to_model=torch.ones(64, 3, 32, 32))
# write.close()

"""---------------------损失函数 Loss function及优化器--------------------"""
# # 标准差
# Inputs = torch.tensor([1, 2, 3], dtype=torch.float32).reshape([-1, 1, 1, 3])
# Targets = torch.tensor([1, 2, 5], dtype=torch.float32).reshape([-1, 1, 1, 3])
# Loss = L1Loss()
# result = Loss(Inputs, Targets)
# print(result)
#
# # 平方差
# Loss_MSE = MSELoss()
# result = Loss_MSE(Inputs, Targets)
# print(result)

# # 交叉熵损失函数
# datasets = torchvision.datasets.CIFAR10("D:\\PyTorch_Python\\BCI_DataSets", train=True,
#                                         transform=transforms.ToTensor(), download=True)
# dataloader = DataLoader(datasets, 1024)
#
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.model = Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             Flatten(),
#             nn.Linear(in_features=1024, out_features=64),
#             nn.Linear(in_features=64, out_features=10)
#         )
#
#     def forward(self, train):
#         train = self.model(train)
#         return train
#
#
# loss = CrossEntropyLoss()
# My_model = Model().cuda()
#
# # 优化器，随机梯度下降
# optin = torch.optim.SGD(My_model.parameters(), lr=0.01)
#
# # 训练
# t1 = time.time()
# for i in range(20):
#     running_loss = []
#     for img, target in dataloader:
#         img = img.cuda()
#         target = target.cuda()
#         # print('img_shape:', img.shape, 'target:', target.shape)  # 观察数据结构
#         Output = My_model(img)
#         result_loss = loss(Output, target)
#         optin.zero_grad()  # 梯度调为0
#         result_loss.backward()  # 误差反向传播
#         optin.step()  # 优化
#         running_loss.append(result_loss)
#     print(f'训练次数:{i + 1}', running_loss)
# t2 = time.time()
# print((t2 - t1) * 1000 / 360)

"""---------------------现有模型修改--------------------"""
# train_data = torchvision.datasets.CIFAR10(r'D:\PyTorch_Python\BCI_DataSets', train=True,
#                                           transform=transforms.ToTensor(), download=True)
# test_data = torchvision.datasets.CIFAR10(r'D:\PyTorch_Python\BCI_DataSets', train=False,
#                                          transform=transforms.ToTensor(), download=True)
#
# # train = DataLoader(train_data, 64)
# # test = DataLoader(test_data, 64)
# #
# # class VGG_Model(nn.Module):
#
# vgg16_False = torchvision.models.vgg16(pretrained=False)
# vgg16_True = torchvision.models.vgg16(pretrained=True)
# # vgg16_True.add_module('Add_linear', nn.Linear(1000, 10))  # 增加到最后
# vgg16_True.classifier.add_module('Add_linear', nn.Linear(1000, 10))  # 添加到classifier块中
# print(vgg16_True)
# vgg16_False.classifier[6] = nn.Linear(4096, 10)  # 修改现有层
# print(vgg16_False)

"""---------------------模型保存、加载--------------------"""
# vgg16_False = torchvision.models.vgg16(pretrained=False)
# # 保存方式1：保存模型及参数
# torch.save(vgg16_False, 'vgg16_method1.pth')
#
# # 加载方式1
# model = torch.load('vgg16_method1.pth')

# （官方推荐）保存方式2：将模型状态（参数）保存为字典形式
# torch.save(vgg16_False.state_dict(), 'vgg16_method2.pth')
# 加载方式2
# model = torch.load('vgg16_method2.pth')
# print(model)
#
# vgg16 = torchvision.models.vgg16(pretrained=False)  # 定义网络
# vgg16.load_state_dict(torch.load('vgg16_method2.pth'))  # 加载状态
# print(vgg16)


# # 陷阱，在其他文件加载模型时需要将原始模型在文件中再定义一次，但无需实例化模型
# class mymodel(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 64, 3),
#             nn.MaxPool2d(2),
#             nn.Linear(64, 10))
#
#     def forward(self, Imput):
#         output = self.model(Imput)
#         return output
#
#
# model = mymodel()
# torch.save(model, 'mymodel.pth')  # 保存

"""---------------------采用GPU训练--------------------"""
# # 方式1：调用模型、数据、损失函数的.cuda()
# import torch
# import torchvision
# from torch.utils.data import DataLoader
# from torchvision.transforms import transforms
# from torch.utils.tensorboard import SummaryWriter
#
#
# # 模型
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.model = Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             Flatten(),
#             nn.Linear(in_features=1024, out_features=64),
#             nn.Linear(in_features=64, out_features=10)
#         )
#
#     def forward(self, train):
#         train = self.model(train)
#         return train
#
#
# # 测试网络正确性
# if __name__ == '__main__':
#     train_model = Model()
#     Input = torch.ones([64, 3, 32, 32])
#     Output = train_model(Input)
#     print(Output.shape)
#     # 返回每张图片为各个标签的概率
#
# # 训练
# writer = SummaryWriter('CIFAR-10')
# # 下载数据集
# data_train = torchvision.datasets.CIFAR10(r'D:\PyTorch_Python\BCI_DataSets', train=True,
#                                           transform=transforms.ToTensor(), download=True)
# data_test = torchvision.datasets.CIFAR10(r'D:\PyTorch_Python\BCI_DataSets', train=False,
#                                          transform=transforms.ToTensor(), download=True)
# print(len(data_train), len(data_test))
# print(data_train, data_test)
# # 加载数据
# train = DataLoader(data_train, batch_size=64)
# test = DataLoader(data_test, batch_size=64)
# # 创建模型
# My_Model = Model().cuda()  # 调用GPU
# # 创建损失函数
# loss_fc = nn.CrossEntropyLoss().cuda()  # 调用GPU
# # 定义优化器
# learning_rate = 0.01
# optinizer = torch.optim.SGD(My_Model.parameters(), lr=learning_rate)
#
# # 训练网络参数设置
# start_time = time.time()
# train_step = 0
# test_step = 0
# epoch = 10  # 训练轮次
#
# # 模型训练主体
# for i in range(epoch):
#     print(f'--------训练{(i + 1)}第次---------')
#     # 开始训练
#     My_Model.train()  # 训练
#     for img, target in train:
#         train_output = My_Model(img.cuda())  # 调用GPU
#         loss = loss_fc(train_output, target.cuda())  # 计算损失值
#         # 模型优化
#         optinizer.zero_grad()  # 梯度清零
#         loss.backward()  # 误差反向
#         optinizer.step()  # 优化
#         train_step += 1
#         if train_step % 100 == 0:
#             print('训练次数：', train_step, '损失值：', loss.item())
#             writer.add_scalar('train_loss', loss.item(), train_step)  # 绘图表示
#
#     # 检验模型训练结果（用训练后的模型识别测试集）
#     test_loss = 0
#     total_accuracy = 0
#     My_Model.eval()  # 测试
#     with torch.no_grad():  # 不再对模型调优
#         for img, target in test:
#             test_output = My_Model(img.cuda())
#             loss = loss_fc(test_output, target.cuda())  # 计算损失值
#             test_loss += loss.item()
#             accuracy = (test_output.argmax(1) == target.cuda()).sum()
#             total_accuracy += accuracy
#     print('测试损失值：', test_loss)
#     print('整体测似精度：', total_accuracy / len(data_train))
#     writer.add_scalar('test_loss', test_loss, test_step)  # 绘图表示
#     test_step += 1
#
#     # 保存每一轮训练后的模型
#     torch.save(Model, 'My_Model_{}.pth'.format(i))
#
# end_time = time.time()
# print('程序用时：', end_time - start_time)
# writer.close()
#
# # 方式2：to
# device = torch.device('cuda')  # 定义训练设备
# My_Model = My_Model.to(device)  # 网络放入设备

"""---------------------深度ResNet--------------------"""
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.Resize(40)
                                   , transforms.RandomHorizontalFlip()  # 随机水平翻转图片
                                   , transforms.RandomCrop(32)
                                   , transforms.ToTensor()
                                ])

train_datasets = dsets.CIFAR10(root=r'D:\PyTorch_Python\Dataset_packing', train=True
                               , transform=transform, download=True)
tset_datasets = dsets.CIFAR10(root=r'D:\PyTorch_Python\Dataset_packing', train=False
                              , transform=transform, download=True)

train_loader = DataLoader(train_datasets, batch_size=128, shuffle=False)
test_loader = DataLoader(tset_datasets, batch_size=128, shuffle=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# 残差块
class ResidualBlack(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 判断残差块是否需要发挥作用
        if self.downsample:
            
            x = self.downsample(x)
        out += x  # 比较有无残差快内卷积部分效果哪个好
        out = self.relu(out)
        return out


# 构建残差网络
class Resnet(nn.Module):
    def __init__(self):
        super().__init__()


"""---------------------循环神经网络、长短时记忆--------------------"""
# """RNN MNIST手写字母识别"""
# # Import packages
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torchvision.datasets as dst
# import torchvision.transforms as transforms
# from torch.optim import Adam
# import torchvision as tv
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Variables setting
# sequence_length = 28
# input_size = 28
# hidden_size = 128
# num_classes = 10
# num_layers = 2
# batch_size = 100
# num_epochs = 50
# lr = 0.001
#
# # Download MNIST datasets
# trainmnist = dst.MNIST(root=r'D:\PyTorch_Python\Dataset_packing', train=True
#                             , download=True, transform=transforms.ToTensor())
# testmnist = dst.MNIST(root=r'D:\PyTorch_Python\Dataset_packing', train=False
#                             , download=True, transform=transforms.ToTensor())
#
# # Loader data
# train_loader = DataLoader(trainmnist, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(testmnist, batch_size=batch_size, shuffle=True)
#
#
# # Create RNN structure
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(RNN, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size
#                             , num_layers=num_layers
#                             , batch_first=True
#                             )
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, input):
#         output, _ = self.LSTM(input)
#         output = output[:, -1, :]  # 只保留每个样本最后一个序列数据
#         output = self.fc(output)
#         return output
#
#
# # Train model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = RNN(input_size, hidden_size, num_layers, num_classes)
# optimizer = Adam(model.parameters(), lr=lr)
# loss_func = nn.CrossEntropyLoss()
# model = model.to(device)
# print(model)
# model.train()
# for epoch in range(num_epochs):
#     total_loss = 0
#     for iteration, data in enumerate(train_loader):
#         image, target = data
#         # print(image.shape, target.shape)
#         image = image.squeeze()  # LSTM接收最高3D输入，压缩颜色通道，第一个28为序列长度，第二个28为每个序列数据长度
#         output = model(image.to(device))
#
#         optimizer.zero_grad()
#         loss = loss_func(output, target.to(device))
#         loss.backward()
#         optimizer.step()
#
#         if iteration % 100 == 0:
#             total_loss += loss.item()
#             print("Epoch {}, Iteration {},  Current loss values {}".format(epoch, iteration, total_loss))
