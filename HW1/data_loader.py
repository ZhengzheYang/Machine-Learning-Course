import numpy as np


class DataLoader:
    def __init__(self):
        self.trn_0 = []
        self.trn_1 = []
        self.tst_0 = []
        self.tst_1 = []
        self.trn_all = []
        self.tst_all = []

    def load(self):
        test = open("HW_1_testing.txt")
        train = open("HW_1_training.txt")

        #process the test and train data
        for line in test:
            line_split = line.split()
            if line_split[2] == "0":
                self.tst_0.append([float(i) for i in line_split])
            elif line_split[2] == "1":
                self.tst_1.append([float(i) for i in line_split])
            else:
                continue
            self.tst_all.append([float(i) for i in line_split])

        for line in train:
            line_split = line.split()
            if line_split[2] == "0":
                self.trn_0.append([float(i) for i in line_split])
            elif line_split[2] == "1":
                self.trn_1.append([float(i) for i in line_split])
            else:
                continue
            self.trn_all.append([float(i) for i in line_split])

        self.trn_0 = np.array(self.trn_0).T
        self.trn_1 = np.array(self.trn_1).T
        self.tst_0 = np.array(self.tst_0).T
        self.tst_1 = np.array(self.tst_1).T
        self.trn_all = np.array(self.trn_all).T
        self.tst_all = np.array(self.tst_all).T