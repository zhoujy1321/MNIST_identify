import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from lenet5 import LeNet5

model_path = './model.pth'
model = LeNet5()
model.load_state_dict(torch.load(model_path))
# model = torch.load('all_model.pkl')


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # 把模型转为test模式

    img = cv2.imread('./test4_9.png', 0)  # 以灰度图的方式读取要预测的图片
    img = cv2.resize(img, (28, 28))

    height, width = img.shape
    dst = np.zeros((height, width), np.uint8)
    for i in range(height):     # 反色处理，将自己手写的白底黑字图片转化为和数据集里相同的黑底白字图片
        for j in range(width):
            dst[i, j] = 255 - img[i, j]
    img = dst
    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]
    img = torch.from_numpy(img)
    output = model(Variable(img))
    output = F.softmax(output, dim=1)
    output = Variable(output)
    # output = output.numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    print(output)  # 10个数字可能的概率
    result = np.argmax(output)  # 选出概率最大的一个
    print(result.item())


if __name__ == "__main__":
    main()
