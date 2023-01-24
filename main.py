import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import mnist_dataset
from lenet5 import LeNet5

'''
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=100, shuffle=False)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=100, shuffle=False)
'''
# 参数初始化
learning_rate = 0.0005   # 学习率
weight_decay = 5e-4  # 权重衰减
momentum = 0.9  # 冲量
max_epoch = 15
train_batchsize = 100
test_batchsize = 100
endpoint_path = 'C:/Users/asus/Desktop/mnist手写识别/model_state.pkl'
# dataset and dataloader 初始化
train_set = mnist_dataset(train=True)
test_set = mnist_dataset(train=False)
train_loader = DataLoader(train_set, batch_size=train_batchsize, shuffle=True)  # 加载数据
test_loader = DataLoader(test_set, batch_size=test_batchsize, shuffle=True)
# 网络初始化
net = LeNet5()
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


def train_epoch(epoch):
    net.train()
    epoch_loss = 0
    tqdm_loader = tqdm(train_loader, total=len(train_loader))
    for idx, (data, target) in enumerate(tqdm_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()  # 梯度初始化为0
        output = net(data)  # 训练后的结果
        output = F.log_softmax(output, dim=1)  # 计算分类后每个数字的概率值
        loss = criterion(output, target.long())
        epoch_loss += loss.item()
        loss.backward()  # 反向传播
        optimizer.step()  # 参数优化
    print('\n Train epoch : {} \t Loss : {:.6f}'.format(epoch, epoch_loss / len(train_loader)))

    return epoch_loss / len(train_loader)


def valid():
    net.eval()
    accuracy = 0  # 正确率
    epoch_loss = 0  # 测试损失
    tqdm_loader = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm_loader):
            data, target = Variable(data), Variable(target)
            output = net(data)  # 测试数据
            output = F.log_softmax(output, dim=1)  # 计算损失
            loss = criterion(output, target.long())
            epoch_loss += loss.item()
            predict = output.data.max(dim=1)[1]  # 找到概率值最大的下标
            accuracy += sum((predict - target) == 0).item() / test_batchsize
    print('\n Test: Loss : {:.5f}, Accuracy : {:.5f} \n'.format(epoch_loss / len(test_loader),
                                                                accuracy / len(test_loader)))
    return epoch_loss / len(test_loader), accuracy / len(test_loader)


def main():
    train_loss = []
    test_loss = []
    test_acc = []
    for epoch in range(max_epoch):
        loss = train_epoch(epoch)
        train_loss.append(loss)
        loss, acc = valid()
        test_loss.append(loss)
        test_acc.append(acc)
        torch.save(net.state_dict(), './model_state.pkl')  # 保存模型参数
    # training loss曲线可视化
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(train_loss)), train_loss, color='red')
    ax1.plot(range(len(test_loss)), test_loss, color='green')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.legend(['Train Loss', 'Test Loss'])
    ax2 = ax1.twinx()
    ax2.plot(range(len(test_acc)), test_acc, color='blue')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Training epochs')
    ax2.legend(['Accuracy'])
    # plt.title('RMSprop,lr=0.005,epoch=15')
    plt.title('lr=0.0005,momentum=0.9,epoch=15')
    plt.savefig('training_loss_lr=0.0005.png', bbox_inches='tight')
    # plt.savefig('training_loss_RMSprop_lr=0.005.png', bbox_inches='tight')


if __name__ == "__main__":
    main()
