import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import DataHelper

class RandomForest:
    def __init__(self, x_train, y_train, x_test, y_test, attr_list, num_trees=None, stabilization=0.002):
        self.num_trees = num_trees
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.attr_list = attr_list
        self.oob_error_rate = []
        self.selected_num_trees = None
        self.stabilization = stabilization

    def fit(self):
        for i in self.num_trees:
            rf = RandomForestClassifier(n_estimators=i, oob_score=True)
            rf.fit(self.x_train, self.y_train)
            self.oob_error_rate.append(1 - rf.oob_score_)
            print("The OOB error rate for %s trees is: %.4f" % (i, 1 - rf.oob_score_))

        self.plot(self.num_trees, self.oob_error_rate)

    def plot(self, num_trees, oob_error_rate):
        plt.figure()
        plt.xlabel("Number of trees")
        plt.ylabel("OOB error rate")
        plt.title("OOB error rate v.s. the number of trees")
        plt.plot(num_trees, oob_error_rate)

    def select(self):
        target = self.oob_error_rate[-1]

        for i, error in enumerate(self.oob_error_rate):
            if np.abs(error - target) <= self.stabilization:
                self.selected_num_trees = (i, error)
                return

    def fit_with_selected(self):
        index, error = self.selected_num_trees
        num_trees = self.num_trees[index]
        rf = RandomForestClassifier(n_estimators=num_trees)
        rf.fit(self.x_train, self.y_train)

        print("The error rate for the random forest classifier with %s estimators is %.4f"
              % (num_trees, 1 - rf.score(self.x_test, self.y_test)))

        self.rank(rf.feature_importances_)

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
    # print(attr_list)
    rf = RandomForest(x_train, y_train, x_test, y_test, attr_list, num_trees=np.arange(10, 510, 10), stabilization=0.002)
    rf.fit()
    rf.select()
    rf.fit_with_selected()
    plt.show()

if __name__ == '__main__':
    main()