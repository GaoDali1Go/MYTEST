from __future__ import print_function
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import warnings
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import torch
import torch.nn as nn
from torchvision.models import alexnet
from torchvision import models
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

class MyDataset(Dataset):  # 继承Dataset
    def __init__(self, root_path, dir, transform=None):  # 初始化一些属性
        self.path_dir = root_path + dir  # 文件路径
        self.transform = transform  # 对图形进行处理，如标准化、截取、转换等
        self.images = os.listdir(self.path_dir)  # 把路径下的所有文件放在一个列表中

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回图像及标签
        image_index = self.images[index]  # 根据索引获取图像文件名称
        img_path = os.path.join(self.path_dir, image_index)  # 获取图像的路径或目录
        img = Image.open(img_path).convert('RGB')  # 读取图像

        # 根据目录名称获取图像标签（0 正常 1 难行 2 管道）
        label = img_path.split('\\')[-1].split('.')[0]
        label = int(label.split('_')[1])
        # if label > 0:
        #     label = 1
        if self.transform is not None:
            img = self.transform(img)
        return img, label

transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])  # 标准化至[-1, 1]，规定均值和标准差
         ])

def load_training(root_path, dir, batch_size, kwargs):
    data = MyDataset(root_path, dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    data = MyDataset(root_path, dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader

"""核函数定义"""
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) #/len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


# """调用写好的Alexnet"""

# import torch
# import torch.nn as nn
# from torchvision.models import alexnet


class alex_net(nn.Module):
    def __init__(self, num_classes):
        super(alex_net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096,num_classes)
        )
        self.gategory = nn.Linear(4096, num_classes)

    def forward(self, source, target):
        loss = 0
        source = self.features(source)
        source = self.avgpool(source)
        source = torch.flatten(source, 1)
        source = self.classifier(source)

        if self.training == True:
            target = self.features(target)
            target = self.avgpool(target)
            target = torch.flatten(target, 1)
            target = self.classifier(target)
            loss += mmd_rbf_noaccelerate(source, target)
        #             loss += mmd.mmd_rbf_noaccelerate(source2, target2)
        #             loss += mmd.mmd_rbf_noaccelerate(source3, target3)

        source = self.gategory(source)
        # target =self.gategory(target)

        return source, loss

# print(model)

# "进行训练"
# from __future__ import print_function
# import argparse
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
# import os
# import math
# from torch.utils import model_zoo


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
batch_size = 100
iteration = 200000
lr = 0.0001
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = r"C:\Users\Gao_D\Desktop\博士工作\科研-迁移学习\数据"
src_name = r"\10.23-26图像"
tgt_name = r"\11.6图像"

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

src_loader = load_training(root_path, src_name, batch_size, kwargs)
tgt_train_loader = load_training(root_path, tgt_name, batch_size, kwargs)
tgt_test_loader = load_testing(root_path, tgt_name, batch_size, kwargs)

src_dataset_len = len(src_loader.dataset)
tgt_dataset_len = len(tgt_test_loader.dataset)
src_loader_len = len(src_loader)
tgt_loader_len = len(tgt_train_loader)


def train(model):
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_train_loader)
    correct = 0
    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))
        optimizer = torch.optim.SGD([
            {'params': model.features.parameters()},
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE},
            {'params': model.gategory.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        try:
            tgt_data, _ = tgt_iter.next()
        except Exception as err:
            tgt_iter = iter(tgt_train_loader)
            tgt_data, _ = tgt_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()
            tgt_data = tgt_data.cuda()

        optimizer.zero_grad()
        src_pred, mmd_loss = model(src_data, tgt_data)
        cls_loss = F.nll_loss(F.log_softmax(src_pred, dim=1), src_label)
        lambd = 2 / (1 + math.exp(-10 * (i) / iteration)) - 1
        loss = cls_loss + 15 * lambd * mmd_loss
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item()))

        if i % (log_interval * 20) == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
                src_name, tgt_name, correct, 100. * correct / tgt_dataset_len))


def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in tgt_test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_pred, mmd_loss = model(tgt_test_data, tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()

    test_loss /= tgt_dataset_len
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        tgt_name, test_loss, correct, tgt_dataset_len,
        100. * correct / tgt_dataset_len))
    return correct


if __name__ == '__main__':
    model = alex_net(num_classes=3)
    model_dict = model.state_dict()
    pretrained_model = alexnet(pretrained=True)
    pretrained_dict = pretrained_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for m in model.classifier:
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0,std=0.2)
            m.bias.data.normal_(mean=0, std=0.2)
    model.gategory.weight.data.normal_(mean=0,std=0.2)  # 全连接层参数初始化
    model.gategory.bias.data.normal_(mean=0, std=0.2)
    model_dict = model.state_dict()
    model.load_state_dict(model_dict)
    print(model)
    if cuda:
        model.cuda()
    train(model)
