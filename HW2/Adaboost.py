import numpy as np
import DataHelper

class DecisionRule:
    def greedy(self, x_train, y_train, weights, step):
        self.x_train = x_train
        self.y_train = y_train
        self.weights = weights

        separation_x1 = self.findAllCutoffs(np.sort(x_train[:, 0]))
        separation_x2 = self.findAllCutoffs(np.sort(x_train[:, 1]))
        # print(separation_x1)

        best_rate = float('inf')
        best_separation = None

        for x1 in separation_x1:
            current_separation = (x1, 'x1')
            error = self.evaluate(self.x_train, self.y_train, current_separation)
            if error < best_rate:
                best_rate = error
                best_separation = current_separation

        for x2 in separation_x2:
            current_separation = (x2, 'x2')
            error = self.evaluate(self.x_train, self.y_train, current_separation)
            if error < best_rate:
                best_rate = error
                best_separation = current_separation
        self.best_separation = best_separation
        # print(best_separation)
        return best_separation

    def findAllCutoffs(self, data):
        res = []
        for i in range(1, len(data)):
            res.append((data[i] + data[i - 1]) / 2)
        return res

    def separate(self, current_separation, x_data):
        x_cur = current_separation[0]
        res = []
        # count_1 = 0
        # count_neg_1 = 0
        direction = 0 if current_separation[1] == 'x1' else 1
        # print(direction)
        # for i, x in enumerate(self.x_train[:, direction]):
        #     if x < x_cur:
        #         if self.y_train[i] == 1:
        #             count_1 += 1
        #         else:
        #             count_neg_1 += 1
        # if count_1 > count_neg_1:
        for x in x_data[:, direction]:
            if x < x_cur:
                res.append(-1)
            else:
                res.append(1)
        # else:
        #     for x in x_data[:, direction]:
        #         if x < x_cur:
        #             res.append(-1)
        #         else:
        #             res.append(1)
        return res

    def evaluate(self, x_data, y_data, current_separation):
        predicted = self.separate(current_separation, x_data)
        weighted_sum = 0.0
        # count = 0.0
        for i, y in enumerate(predicted):
            if y != y_data[i]:
                weighted_sum += self.weights[i]
        return weighted_sum


class Adaboost:
    def __init__(self, weights, epochs, decision):
        self.weights = weights
        # print(self.weights)
        self.alpha_m = []
        self.separations = []
        self.epochs = epochs
        self.decision = decision


    def train(self, x_train, y_train):
        for epoch in range(self.epochs):
            separation = self.decision.greedy(x_train, y_train, self.weights, 0.001)
            predicted = self.decision.separate(separation, x_train)
            self.separations.append(self.decision.best_separation)
            weighted_sum = 0.0
            for i, prediction in enumerate(predicted):
                if prediction != y_train[i]:
                    weighted_sum += self.weights[i]
            err_m = weighted_sum / sum(self.weights)
            alpha_m = np.log((1 - err_m) / err_m)
            self.alpha_m.append(alpha_m)
            for i, w in enumerate(self.weights):
                if predicted[i] != y_train[i]:
                    self.weights[i] *= np.exp(alpha_m)


    def predict(self, x_test):
        res = np.zeros(len(x_test))
        for epoch in range(self.epochs):
            # print(self.classifiers[epoch].weights)
            for i, x in enumerate(np.multiply(self.alpha_m[epoch], self.decision.separate(self.separations[epoch], x_test))):
                res[i] += x
        res = [1 if x >= 0.0 else -1 for x in res]
        return res

    def evaluate(self, predictions, y_test):
        count = 0.0
        for prediction, y in zip(predictions, y_test):
            if prediction != y:
                count += 1
        return (count / len(y_test) * 100)

def main():
    data = DataHelper.Data()
    x_train, y_train, x_test, y_test, _ = data.loadData("hw2_data_1.txt", 2, 70)

    epoch = 3
    weights = np.ones(len(x_train)) / len(x_train)
    adaboost = Adaboost(weights, epoch, DecisionRule())
    adaboost.train(x_train, y_train)
    prediction = adaboost.predict(x_test)
    print("Error rate for %d iterations is %.2f%%" % (epoch, adaboost.evaluate(prediction, y_test)))

    epoch = 5
    weights = np.ones(len(x_train)) / len(x_train)
    adaboost = Adaboost(weights, epoch, DecisionRule())
    adaboost.train(x_train, y_train)
    prediction = adaboost.predict(x_test)
    print("Error rate for %d iterations is %.2f%%" % (epoch, adaboost.evaluate(prediction, y_test)))

    epoch = 10
    weights = np.ones(len(x_train)) / len(x_train)
    adaboost = Adaboost(weights, epoch, DecisionRule())
    adaboost.train(x_train, y_train)
    prediction = adaboost.predict(x_test)
    print("Error rate for %d iterations is %.2f%%" % (epoch, adaboost.evaluate(prediction, y_test)))

    epoch = 20
    weights = np.ones(len(x_train)) / len(x_train)
    adaboost = Adaboost(weights, epoch, DecisionRule())
    adaboost.train(x_train, y_train)
    prediction = adaboost.predict(x_test)
    print("Error rate for %d iterations is %.2f%%" % (epoch, adaboost.evaluate(prediction, y_test)))

if __name__ == "__main__":
    main()
