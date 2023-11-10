import random
import time
import numpy as np
import imageio.v2 as imageio
from BPNetwork import BPNetwork
import pickle
test_dir = "E:/test_data/"
if __name__ == '__main__':
    test_size = 240
    test_input = np.random.random((test_size, 12, 28 * 28))
    output_data = np.zeros((12, 12))
    for i in range(0, test_size):
        for j in range(0, 12):
            test_input[i][j] = list(np.array(imageio.imread(test_dir + str(j+1) + "/" + str(i+1) + ".bmp")).flatten())
    for i in range(0, 12):
        output_data[i][i] = 1
    # 手写字的识别
    print(" test")
    with open('my_model_20_15.pkl', 'rb') as f:
        character = pickle.load(f)
    rightness = 0
    for j in range(0, test_size):
        for k in range(0, 12):
            # test = character.test(test_input[j][k], output_data[k])
            result = character.forward_propagate(test_input[j][k])
            if result.index(max(result)) == k:
                rightness += 1
            # print("测试值：", result.index(max(result)), "实际：", k)
            # print(character.test(test_input[j][k], output_data[k]))
    print("testing rightness:", rightness / 12 / test_size)
    # for i in range(0, 10):
    #     random.seed(time.time())
    #     x = random.randint(0, 11)
    #     y = random.randint(0, 619)
    #     input_data = list(np.array(imageio.imread("train/" + str(x+1) + "/" + str(y+1) + ".bmp")).flatten())
    #     result = character.forward_propagate(input_data)
    #     test = result.index(max(result))
    #     # print(x, y)
    #     match x:
    #         case 0:
    #             print("输入字为:博")
    #         case 1:
    #             print("输入字为:学")
    #         case 2:
    #             print("输入字为:笃")
    #         case 3:
    #             print("输入字为:志")
    #         case 4:
    #             print("输入字为:切")
    #         case 5:
    #             print("输入字为:问")
    #         case 6:
    #             print("输入字为:近")
    #         case 7:
    #             print("输入字为:思")
    #         case 8:
    #             print("输入字为:自")
    #         case 9:
    #             print("输入字为:由")
    #         case 10:
    #             print("输入字为:无")
    #         case 11:
    #             print("输入字为:用")
    #     match test:
    #         case 0:
    #             print("扫描识别结果为:博")
    #         case 1:
    #             print("扫描识别结果为:学")
    #         case 2:
    #             print("扫描识别结果为:笃")
    #         case 3:
    #             print("扫描识别结果为:志")
    #         case 4:
    #             print("扫描识别结果为:切")
    #         case 5:
    #             print("扫描识别结果为:问")
    #         case 6:
    #             print("扫描识别结果为:近")
    #         case 7:
    #             print("扫描识别结果为:思")
    #         case 8:
    #             print("扫描识别结果为:自")
    #         case 9:
    #             print("扫描识别结果为:由")
    #         case 10:
    #             print("扫描识别结果为:无")
    #         case 11:
    #             print("扫描识别结果为:用")
