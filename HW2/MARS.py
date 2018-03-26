import numpy as np
import matplotlib.pyplot as plt
from pyearth import Earth
import DataHelper

class MARS:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.classifier = None

    def fit(self):
        self.classifier = Earth()
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self):
        return self.classifier.predict(self.x_test)

    def dichotomize(self, predictions):
        median = np.median(predictions)
        res = np.array([1 if y >= median else -1 for y in predictions])
        return res

    def evaluate(self):
        predictions = self.dichotomize(self.predict())
        # print(predictions)
        error = 0.0
        for y, correct in zip(predictions, self.y_test):
            if y != correct:
                error += 1
        return error / len(self.y_test)




def main():
    data = DataHelper.Data()
    x_train, y_train, x_test, y_test, _ = data.loadData("hw2_data_2.txt", 20, 700)
    mars = MARS(x_train, y_train, x_test, y_test)
    mars.fit()
    print("The testing error rate for MARS classifier is: %.4f" % mars.evaluate())


if __name__ == '__main__':
    main()