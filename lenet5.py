import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层 1: 28*28 -> 14*14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)  # 1:灰度图片的通道，6:输出通道，5:kernel大小
        self.relu1 = nn.ReLU()  # 激活层,输出是非线性函数,提高神经网络的表达能力
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层,对图片进行压缩;2:kernel大小,2:步长
        # 卷积层 2: 14*14 -> 5*5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1: 16*5*5 -> 120
        self.fc3 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.relu3 = nn.ReLU()
        # 全连接层2: 120 -> 84
        self.fc4 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        # 全连接层3: 84 -> 10
        self.fc5 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)   # 输入:1*28*28,输出:6*28*28(28+4-5+1)
        x = self.relu1(x)   # 保持shape不变,输出:6*28*28
        x = self.max1(x)    # 输出:6*14*14
        x = self.conv2(x)   # 输入:6*14*14,输出:16*10*10(14-5+1)
        x = self.relu2(x)   # 输出:16*10*10
        x = self.max2(x)    # 输出:16*5*5

        x = x.view(-1, 16 * 5 * 5)  # 自动计算维度:16*5*5=400,并拉平
        x = self.fc3(x)     # 输入:400,输出:120
        x = self.relu3(x)
        x = self.fc4(x)     # 输入:120,输出:84
        x = self.relu4(x)
        x = self.fc5(x)     # 输入:84,输出:10
        return x


def main():
    net = LeNet5()
    x = torch.randn([3, 1, 28, 28])
    # print(x)
    print('input:', x.shape)
    out = net(x)
    # print(out)
    print('output:', out.shape)


if __name__ == "__main__":
    main()
