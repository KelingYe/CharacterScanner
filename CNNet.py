import pickle
import torch
import torch.utils.data
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import datasets
# 批训练样本数目
BATCH_SIZE = 120
# 迭代次数
MAX_STEPS = 5
# 学习率
THRESHOLD = 0.001
# 训练集
TRAIN_DIR = "train"
# 样本规格化规则
train_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=(0.5,), std=(0.5,))])
# 获取图片样本
custom_datasets = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
# 规定训练集测试集比例
train_size = int(len(custom_datasets) * 0.8)
test_size = len(custom_datasets) - train_size
train_datasets, test_datasets = torch.utils.data.random_split(custom_datasets, [train_size, test_size])
# 生成训练集测试集迭代器
train_dataloader = torch.utils.data.DataLoader(train_datasets,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_datasets,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=2)
# 将数据载入图形处理器加快运算速率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNet(nn.Module):
    # 卷积神经网络结构搭建
    def __init__(self):
        super(CNNet, self).__init__()
        # 特征层
        self.features = nn.Sequential(
            # 卷积层
            # 输入图片为三通道图，故输入通道为3
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            # 批量归一化
            nn.BatchNorm2d(num_features=32),
            # 激活函数
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 最大池化层
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 分类层
        self.classifier = nn.Sequential(
            # dropout层
            nn.Dropout(p=0.5),
            nn.Linear(64*7*7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 12),
        )

    # 向前传递
    def forward(self, x):
        x = self.features(x)
        # 展开为一维向量
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 训练模型
def train(epoch_num, _model, _device, _train_loader, _optimizer, _lr_scheduler):
    _model.train()
    for epoch in range(epoch_num):
        for i, (images, labels) in enumerate(_train_loader):
            samples = images.to(_device)
            labels = labels.to(_device)
            output = _model(samples.reshape(-1, 3, 28, 28))
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("epoch:{}/{}, steps:{}, loss:{:.4f}".format(epoch + 1, epoch_num, i + 1, loss.item()))
        _lr_scheduler.step()


# 测试模型
def test(_test_loader, _model, _device):
    _model.eval()
    loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in _test_loader:
            data, target = data.to(_device), target.to(_device)
            output = _model(data.reshape(-1, 3, 28, 28))
            loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(_test_loader.dataset)

    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))


if __name__ == '__main__':
    ConvModel = pickle.load(open('conv.pkl', 'rb'))
    # ConvModel = CNNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(ConvModel.parameters(), lr=THRESHOLD)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    for epoch in range(1, MAX_STEPS + 1):
        train(epoch, ConvModel, DEVICE, train_dataloader, optimizer, exp_lr_scheduler)
        pickle.dump(ConvModel, open('conv.pkl', 'wb'))
        test(test_dataloader, ConvModel, DEVICE)
