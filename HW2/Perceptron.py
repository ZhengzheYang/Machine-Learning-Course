import numpy as np
import DataHelper

class perceptron():
    def __init__(self, weight, learningRate, epoch):
        self.weights = weight
        self.learningRate = learningRate
        self.epochs = epoch

    # Prediction function. 1 if positive and -1 if negative
    def predict(self, tuple):
        activation = self.weights[0]
        activation += np.sum(self.weights[1:] * tuple)
        return 1.0 if activation >= 0.0 else -1.0

    # Training function. The weights are updated based on the wrong predictions
    def train(self, x_train, label_train):
        for epoch in range(self.epochs):
            for tuple, y in zip(x_train, label_train):
                prediction = self.predict(tuple)    #predict
                if y != prediction:
                    self.weights[0] += self.learningRate * y    #update bias
                    for i, weight in enumerate(self.weights[1:]):
                        self.weights[i + 1] += self.learningRate * y * tuple[i]     # update weights

    # Evaluation function. See how we are doing
    def evaluate(self, x_test, label_test):
        count = 0.0
        for tuple, y in zip(x_test, label_test):
            if self.predict(tuple) != y:
                count += 1
        return (count / len(label_test) * 100)

def main():
    data = DataHelper.Data()
    x_train, y_train, x_test, y_test, _ = data.loadData("hw2_data_1.txt", 2, 70)    # load data
    weights = np.ones(x_train.shape[1] + 1)
    epochs = 50
    model = perceptron(weights, learningRate=1, epoch=epochs)
    model.train(x_train, y_train)
    print("The error rate for perceptron after %i epochs is %.2f %%" % (epochs, model.evaluate(x_test, y_test)))

if __name__ == "__main__":
    main()
