import numpy as np
import imageio.v2 as imageio
from BPNetwork import BPNetwork
import pickle

if __name__ == '__main__':
    # 手写字的识别
    print(" test")
    sample_size = 620
    # test_size = 20 - sample_size
    input_data = np.random.random((sample_size, 12, 28*28))
    output_data = np.zeros((12, 12))
    # test_input = np.random.random((test_size, 12, 28*28))
    # 导入图片
    for i in range(0, sample_size):
        for j in range(0, 12):
            input_data[i][j] = list(np.array(imageio.imread("train/" + str(j+1) + "/" + str(i+1) + ".bmp")).flatten())

    # for i in range(0, test_size):
    #     for j in range(0, 12):
    #         test_input[i][j] = list(np.array(imageio.imread("train/" + str(j+1) + "/" + str(i+1+sample_size) + ".bmp")).flatten())

    for i in range(0, 12):
        output_data[i][i] = 1

    # 初始化网络
    # character = BPNetwork()
    # character.setup(28*28, 12, [20, 15])

    # 载入已训练网络
    with open('my_model_20_15_plus.pkl', 'rb') as f:
        character = pickle.load(f)

    # 打印隐藏层设置和学习率
    hidden = []
    for i in range(len(character.hidden_ns)):
        hidden.append(character.hidden_ns[i]-1)
    print("隐藏层各层神经元数：", hidden)
    threshold = 0.0005
    print("学习率: ", threshold)
    max_steps = 10
    right = 0
    right_p = 0.875
    # 开始训练
    for i in range(0, max_steps):
        epoch = 0.0
        # if i % 10 == 1:
        #     threshold = 0.05
        # threshold /= 1.5
        for j in range(0, sample_size):
            for k in range(0, 12):
                epoch += character.back_propagate_c(input_data[j][k], output_data[k], threshold)
        right = epoch/12/sample_size
        print("epoch:", i + 1, "times", "training rightness:", right)
        # 存入网络
        if right > right_p:
            right_p = right
            with open('my_model_20_15_plus.pkl', 'wb') as f:
                pickle.dump(character, f)
        if right > 0.95:
            break

    # # 测试正确率
    # rightness = 0
    # for j in range(0, test_size):
    #     for k in range(0, 12):
    #         # test = character.test(test_input[j][k], output_data[k])
    #         result = character.forward_propagate(test_input[j][k])
    #         if result.index(max(result)) == k:
    #             rightness += 1
    #         # print("测试值：", result.index(max(result)), "实际：", k)
    #         # print(character.test(test_input[j][k], output_data[k]))
    # print("testing rightness:", rightness/12/test_size)




