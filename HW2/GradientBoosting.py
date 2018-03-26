import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import DataHelper

class GradientBoosting:
    def __init__(self, x_train, y_train, x_test, y_test, attr_list):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.attr_list = attr_list

    def fit(self):
        gb = GradientBoostingClassifier(loss='deviance')
        gb.fit(self.x_train, self.y_train)
        print("The testing error rate for the gradient boosting classifier is %.4f"
              % (1 - gb.score(self.x_test, self.y_test)))
        self.rank(gb.feature_importances_)

    def rank(self, importances):
        sorted_args = np.argsort(importances)[::-1]
        importances = np.array(importances[sorted_args])
        self.attr_list = np.array(self.attr_list[sorted_args])
        print("Rank the variables based on their importance: ")
        for x, importance in zip(self.attr_list, importances):
            print(x, importance)

def main():
    data = DataHelper.Data()
    x_train, y_train, x_test, y_test, attr_list = data.loadData("hw2_data_2.txt", 20, 700)
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, attr_list.shape)
    gb = GradientBoosting(x_train, y_train, x_test, y_test, attr_list)
    gb.fit()

if __name__ == '__main__':
    main()