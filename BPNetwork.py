import random
import numpy as np
import matplotlib.pyplot as plt


random.seed(0)


def rand(a, b):
    return (b-a) * random.random() + a


def generate_w(m, n):
    # 在相邻层的神经元之间拉weight线
    w = [0.0]*m
    for i in range(m):
        w[i] = [0.0]*n
        for j in range(n):
            w[i][j] = rand(-1, 1)
    return w


def generate_b(m):
    b = [0.0]*m
    for i in range(m):
        b[i] = rand(-1, 1)
    return b


def fit_function(x, deriv=False):
    if deriv:
        return 1 - np.tanh(x)*np.tanh(x)
    return np.tanh(x)


class BPNetwork:
    def __init__(self):
        # 输入层神经元个数
        # 拟合sin(x)需要input_n = 2
        # 一个是输入数据，另一个用于调节bias
        self.input_n = 0
        # 输入层神经元输入的数据
        # 包括输入数据和bias（默认为1）
        self.input_cells = []
        # 输入层到隐藏层第一层的weight
        self.input_w = []
        # 输出层神经元个数
        self.output_n = 0
        # 输出层神经元输出值
        self.output_cells = []
        # 隐藏层最后一层到输出层的weight
        self.output_w = []
        # 输出层的bias
        self.output_b = []
        # 输出层的error关于w的偏导
        self.output_deltas = []
        # 隐藏层设置
        # 长度代表隐藏层个数，每个元素代表该层神经元个数
        self.hidden_ns = []
        # 隐藏层weight
        self.hidden_ws = []
        # 每个元素为该层设置的bias值
        self.hidden_bs = []
        # 每层隐藏层的输出值
        self.hidden_results = []
        # error关于i到i+1层weight的偏导
        self.hidden_deltases = []

    def setup(self, input_n, output_n, hidden_set):
        # input_n是输入参数的个数，不等同于神经元的个数
        # +1是新增一个神经元来调节bias
        # 且这个神经元的输入值记为1
        self.input_n = input_n + 1
        self.output_n = output_n
        self.hidden_ns = [0.0]*len(hidden_set)
        for i in range(len(hidden_set)):
            self.hidden_ns[i] = hidden_set[i] + 1
        # 初始化神经元个数列表
        self.input_cells = [1.0]*self.input_n
        self.output_cells = [1.0]*self.output_n
        # 初始化输入层和隐藏层第一层之前的weight
        self.input_w = generate_w(self.input_n, self.hidden_ns[0])
        # 初始化隐藏层之间的weight
        self.hidden_ws = [0.0]*(len(self.hidden_ns)-1)
        for i in range(len(self.hidden_ns)-1):
            self.hidden_ws[i] = generate_w(self.hidden_ns[i], self.hidden_ns[i+1])
        # 初始化隐藏层最后一层到输出层weight
        self.output_w = generate_w(self.hidden_ns[len(self.hidden_ns)-1], self.output_n)
        self.output_b = generate_b(self.output_n)
        self.hidden_bs = [0.0]*len(self.hidden_ns)
        for i in range(len(self.hidden_ns)):
            self.hidden_bs[i] = generate_b(self.hidden_ns[i])
        self.hidden_results = [0.0]*(len(self.hidden_ns))

    def forward_propagate(self, input_):
        # 向前传播
        for i in range(len(input_)):
            self.input_cells[i] = input_[i]
        # 输入层
        self.hidden_results[0] = [0.0]*self.hidden_ns[0]
        for h in range(self.hidden_ns[0]):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_w[i][h] * self.input_cells[i]
            self.hidden_results[0][h] = fit_function(total+self.hidden_bs[0][h])
        # 隐藏层
        for k in range(len(self.hidden_ns)-1):
            self.hidden_results[k+1] = [0.0]*self.hidden_ns[k+1]
            for h in range(self.hidden_ns[k+1]):
                total = 0.0
                for i in range(self.hidden_ns[k]):
                    total += self.hidden_ws[k][i][h] * self.hidden_results[k][i]

                self.hidden_results[k+1][h] = fit_function(total+self.hidden_bs[k+1][h])
        # 输出层
        for h in range(self.output_n):
            total = 0.0
            for i in range(self.hidden_ns[len(self.hidden_ns)-1]):
                total += self.output_w[i][h] * self.hidden_results[len(self.hidden_ns)-1][i]
                self.output_cells[h] = fit_function(total+self.output_b[h])

        return self.output_cells[:]

    def get_deltas(self, label):
        # 反向传输错误
        self.output_deltas = [0.0]*self.output_n
        # 输出层deltas
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            self.output_deltas[o] = fit_function(self.output_cells[o], True) * error
        # 隐层deltas
        tmp_deltas = self.output_deltas
        tmp_w = self.output_w
        self.hidden_deltases = [0.0]*(len(self.hidden_ns))
        k = len(self.hidden_ns) - 1
        while k >= 0:
            self.hidden_deltases[k] = [0.0]*(self.hidden_ns[k])
            for o in range(self.hidden_ns[k]):
                error = 0.0
                for i in range(len(tmp_deltas)):
                    error += tmp_deltas[i] * tmp_w[o][i]
                self.hidden_deltases[k][o] = fit_function(self.hidden_results[k][o], True) * error
            k = k - 1
            if k >= 0:
                tmp_w = self.hidden_ws[k]
                tmp_deltas = self.hidden_deltases[k+1]
            else:
                break

    def renew_w(self, learn):
        # 更新隐藏层到输出层权重
        k = len(self.hidden_ns) - 1
        for i in range(self.hidden_ns[k]):
            for o in range(self.output_n):
                change = self.output_deltas[o] * self.hidden_results[k][i]
                self.output_w[i][o] += change * learn
        # 更新隐层权重
        while k > 0:
            for i in range(self.hidden_ns[k-1]):
                for o in range(self.hidden_ns[k]):
                    change = self.hidden_deltases[k][o] * self.hidden_results[k-1][i]
                    self.hidden_ws[k-1][i][o] += change * learn
            k = k - 1
        # 更新输入层到隐层权重
        for i in range(self.input_n):
            for o in range(self.hidden_ns[0]):
                change = self.hidden_deltases[0][o] * self.input_cells[i]
                self.input_w[i][o] += change * learn

    def renew_b(self, learn):
        # 更新隐藏层bias
        k = len(self.hidden_bs)-1
        while k >= 0:
            for i in range(self.hidden_ns[k]):
                self.hidden_bs[k][i] = self.hidden_bs[k][i] + learn * self.hidden_deltases[k][i]
            k = k - 1
        # 更新输出层bias
        for o in range(self.output_n):
            self.output_b[o] += self.output_deltas[o] * learn

    def get_loss(self, label, output_cell):
        error = 0.0
        for o in range(len(output_cell)):
            error += 0.5 * (label[o] - output_cell[o]) ** 2 / len(output_cell)
        return error

    def get_rightness(self, label, output_cell):
        rightness = 0
        label = label.tolist()
        if output_cell.index(max(output_cell)) == label.index(max(label)):
            rightness = 1
        return rightness

    def back_propagate_c(self, input_, label, learn):
        self.forward_propagate(input_)
        self.get_deltas(label)
        self.renew_w(learn)
        self.renew_b(learn)
        return self.get_rightness(label, self.output_cells)

    def back_propagate(self, input_, label, learn):
        self.forward_propagate(input_)
        self.get_deltas(label)
        self.renew_w(learn)
        self.renew_b(learn)
        return self.get_loss(label, self.output_cells)

    def test(self, input_data, output_data):
        self.forward_propagate(input_data)
        return self.get_loss(output_data, self.output_cells)


    # def train_sin(self, input_datas, labels, threshold, max_steps):
    #     for k in range(max_steps):
    #         epoch = 0.0
    #         for o in range(len(input_datas)):
    #             epoch += self.back_propagate(input_datas[o], labels[o], threshold)
    #         error = epoch / len(input_datas)
    #         print("第 %d 次迭代：" % (k+1))
    #         print(error)
    #
    #
    # def get_train_sin(self):
    #     input_data = []
    #     output_data = []
    #     x = np.linspace(-np.pi, np.pi, 10)
    #     y = np.sin(x)
    #     for i in range(len(x)):
    #         input_data.append([])
    #         input_data[i].append(x[i])
    #     for i in range(len(y)):
    #         output_data.append([])
    #         output_data[i].append(y[i])
    #     return input_data, output_data

    # def get_test(self):
    #     pass
    #
    # def get_average_loss(self):
    #     pass

    # def test_sin(self):
    #     input_datas, labels = self.get_train_sin()
    #     self.setup(1, 1, [10, 10, 10])
    #     self.train_sin(input_datas, labels, 0.05, 3000)

        # test, test_label = self.get_test()
        # error = self.get_average_loss(test, test_label)
        # print(error)



