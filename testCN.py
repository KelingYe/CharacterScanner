import pickle
import torch
import torch.utils.data
import torch.nn as nn
from CNNet import CNNet
from torchvision import transforms
from torchvision import datasets
# 批训练样本数目
BATCH_SIZE = 120
# 训练集
TEST_DIR = "E:/test_data"
# 样本规格化规则
test_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=(0.5,), std=(0.5,))])
# 获取图片样本
test_datasets = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
# 规定训练集测试集比例
test_dataloader = torch.utils.data.DataLoader(test_datasets,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=2)
# 将数据载入图形处理器加快运算速率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    test(test_dataloader, ConvModel, DEVICE)
