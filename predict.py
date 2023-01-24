import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import mnist_dataset
from lenet5 import LeNet5

model_path = './model_state.pkl'
model = LeNet5()
model.load_state_dict(torch.load(model_path))

test_set = mnist_dataset(train=False)
test_batchsize = 100
test_loader = DataLoader(test_set, batch_size=test_batchsize)
criterion = nn.CrossEntropyLoss()


def test():
    model.eval()
    accuracy = 0
    epoch_loss = 0
    tqdm_loader = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():  # 模型训练
        for idx, (data, target) in enumerate(tqdm_loader):
            data, target = Variable(data), Variable(target)
            output = model(data)  # 训练后的结果
            output = F.log_softmax(output, dim=1)
            loss = criterion(output, target.long())
            epoch_loss += loss.item()
            predict = output.data.max(dim=1)[1]
            accuracy += sum((predict - target) == 0).item() / test_batchsize

    print('Test: Loss: {:.5f}, Accuracy: {:.5f}'.format(epoch_loss / len(test_loader), accuracy / len(test_loader)))
    return epoch_loss / len(test_loader), accuracy / len(test_loader)


def main():
    loss, acc = test()


if __name__ == "__main__":
    main()
