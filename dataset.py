import gzip
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tfs

# mnist data path
data_path = 'C:/Users/asus/Desktop/mnist手写识别/data/'


def read_image(image_path, label_path):
    with gzip.open(label_path) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_path, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image, label


def img_transform(img, label):  # 对图像做处理
    img = img.reshape([28, 28])
    transform = tfs.Compose([tfs.ToTensor(),                        # 将图片转换成tensor格式
                             tfs.Normalize((0.1307,), (0.3081,))    # 标准化，模型出现过拟合现象时，降低模型复杂度
                             ])
    img_tensor = transform(img)
    label_tensor = torch.tensor(np.int8(label))
    return img_tensor, label_tensor


class mnist_dataset(Dataset):
    def __init__(self, train=True, path=data_path, transform=img_transform):
        super(mnist_dataset, self).__init__()
        self.transform = transform
        if train:
            data_path = path + 'train-images-idx3-ubyte.gz'
            label_path = path + 'train-labels-idx1-ubyte.gz'
        else:
            data_path = path + 't10k-images-idx3-ubyte.gz'
            label_path = path + 't10k-labels-idx1-ubyte.gz'
        self.image_list, self.label_list = read_image(data_path, label_path)
        self.len = len(self.image_list)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img = self.image_list[index]
        label = self.label_list[index]
        img, label = self.transform(img, label)
        return img, label


def main():
    set = mnist_dataset(train=True)
    img, lab = set[0]
    print(img.shape)
    print(lab)


if __name__ == "__main__":
    main()
