import numpy as np

class Data:
    def loadData(self, fileName, attr_num, boundary):
        x_train = []
        label_train = []
        x_test = []
        label_test = []
        attr_list = None
        file = open(fileName)
        for i, line in enumerate(file):
            if i == 0:
                attr_list = [x.replace('"', '') for x in line.split()]
                attr_list.pop()
            elif i <= boundary:
                line = [float(x) for x in line.split()]
                x_train.append(line[:attr_num])
                label_train.append(line[attr_num])
            else:
                line = [float(x) for x in line.split()]
                x_test.append(line[:attr_num])
                label_test.append(line[attr_num])
        return np.array(x_train), np.array(label_train), np.array(x_test), np.array(label_test), np.array(attr_list)