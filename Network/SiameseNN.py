"""
该模型根据王鑫荣毕业论文所得，每个网络由三个卷积-池化层组成
"""

import torch
import torch.nn as nn


# 要求经过CBAM后数据尺寸保持不变
class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channels, ratio=4):
        # 继承父类初始化方法
        super(channel_attention, self).__init__()  # python 2.x

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channels // ratio, out_features=in_channels, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view(b, c)
        avg_pool = avg_pool.view(b, c)

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # sigmoid函数权值归一化
        x_maxpool = self.sigmoid(x_maxpool)
        x_avgpool = self.sigmoid(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view(b, c, 1, 1)
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        return outputs


class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)  # 最大池化，dim=1，按列输出

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)  # 平均池化
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)  # 横向拼接

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)  # 卷积运算
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs


cfg = {'conv3': [16, 'M', 32, 'M', 64, 'M']}


class SNN(nn.Module):
    def __init__(self, num_conv):
        super(SNN, self).__init__()
        self.in_channels = 22
        # CBAM
        self.ca1 = channel_attention(self.in_channels)
        self.sa1 = spatial_attention()

        self.feature = self._make_layers(cfg[num_conv])

        # CBAM
        self.ca1 = channel_attention(self.in_channels)
        self.sa1 = spatial_attention()
        self.MLP = nn.Sequential(
            nn.Linear(128, 100)
            , nn.ReLU(inplace=True)
            , nn.Dropout(p=0.5)

            , nn.Linear(100, 100)
            , nn.Sigmoid()
            , nn.Linear(100, 4)
        )

    def forward(self, in_1, in_2):
        out_1 = self.feature(in_1)
        out_1 = out_1.view(out_1.size()[0], -1)
        out_1 = self.MLP(out_1)
        # 输入2
        out_2 = self.feature(in_2)
        out_2 = out_2.view(out_2.size()[0], -1)
        out_2 = self.MLP(out_2)

        return out_1, out_2

    def _make_layers(self, cfg):
        layers = []
        in_channels = 22
        for out_channels in cfg:
            if out_channels == 'M':
                layers += [nn.MaxPool2d(kernel_size=2)]
            else:
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=2)
                    , nn.ReLU(inplace=True)
                    , nn.BatchNorm2d(out_channels)]
                in_channels = out_channels
        return nn.Sequential(*layers)


# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2).reshape(-1, 1)
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                      target * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # 限制最小值为0
        return loss_contrastive


if __name__ == "__main__":
    snn = SNN("conv3")
    # 输入数据结构与卷积核大小需要根据实际计算得到
    output = snn(torch.ones(64, 22, 10, 100), torch.zeros(64, 22, 10, 100))
    print(snn)
