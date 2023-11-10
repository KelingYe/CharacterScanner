import numpy as np

from BPNetwork import BPNetwork


if __name__ == '__main__':
    # sin(x)的拟合
    print(" test")
    t = BPNetwork()
    # 初始化BP网络，输入输出一个神经元，中间层为三层
    t.setup(1, 1, [3])
    # 获得数据
    # threshold为学习率
    # max_steps为epoch次数
    threshold = 0.05
    max_steps = 3000
    input_data = []
    labels = []
    # 生成正弦函数目标值
    x = np.linspace(-np.pi, np.pi, 10)
    y = np.sin(x)
    for i in range(len(x)):
        input_data.append([])
        input_data[i].append(x[i])
    for i in range(len(y)):
        labels.append([])
        labels[i].append(y[i])
    # 开始训练
    for k in range(max_steps):
        epoch = 0.0
        for o in range(len(input_data)):
            epoch += t.back_propagate(input_data[o], labels[o], threshold)
        error = epoch / max_steps
        print("第 %d 次迭代：" % (k + 1))
        print(error)
