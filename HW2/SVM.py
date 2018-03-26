import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import DataHelper

class SVM:
    def __init__(self, x_train, y_train, x_test, y_test, kernel=None, gamma_range=None, degree_range=None):
        self.kernel = kernel
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.gamma_range = gamma_range
        self.degree_range = degree_range

    def run(self):
        if self.kernel == "RADIAL":
            plot_title = "Radial kernel with 10-fold cross-validation v.s. gamma parameters"
            self.plot("gamma", self.gamma_range, self.radialKernel(), plot_title)
        elif self.kernel == "POLYNOMIAL":
            plot_title = "Polynomial kernel with 10-fold cross-validation v.s. degrees"
            self.plot("degree", self.degree_range, self.polynomialKernel(), plot_title)
        elif self.kernel == "SIGMOID":
            plot_title = "Sigmoid kernel with 10-fold cross-validation v.s. gamma parameters"
            self.plot("gamma", self.gamma_range, self.sigmoidKernel(), plot_title)
        else:
            print("Sorry, kernel currently not available. Exit.")
            exit(1)

    def plot(self, x_label, x_axis, error_rate, title):
        plt.figure()
        if x_label == "gamma":
            plt.xscale('log')
        plt.xlabel(x_label)
        plt.ylabel("Error rate")
        plt.title(title)
        plt.plot(x_axis, error_rate)

    def gridSearch(self, param_grid, param):
        grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='accuracy', cv=10)
        fit_result = grid.fit(self.x_train, self.y_train)
        mean_scores = fit_result.cv_results_['mean_test_score']

        error = 1 - mean_scores
        best_gamma = fit_result.best_params_[param]

        return error, best_gamma

    def radialKernel(self):
        grid_dict = [{'kernel': ['rbf'], 'gamma': self.gamma_range}]
        error, best_gamma = self.gridSearch(grid_dict, 'gamma')
        print("The best gamma for the radial kernel is %s" % best_gamma)
        svm = SVC(kernel='rbf', gamma=best_gamma)
        svm.fit(self.x_train, self.y_train)
        print("The best testing error rate for the radial kernel SVM is %.4f" % (1 - svm.score(self.x_test, self.y_test)))
        return error

    def polynomialKernel(self):
        grid_dict = [{'kernel': ['poly'], 'degree': self.degree_range}]
        error, best_degree = self.gridSearch(grid_dict, 'degree')
        print("The best degree for the polynomial kernel is %s" % best_degree)
        svm = SVC(kernel='poly', degree=best_degree)
        svm.fit(self.x_train, self.y_train)
        print("The best testing error rate for the polynomial kernel SVM is %.4f" % (1 - svm.score(self.x_test, self.y_test)))
        return error

    def sigmoidKernel(self):
        grid_dict = [{'kernel': ['sigmoid'], 'gamma': self.gamma_range}]
        error, best_gamma = self.gridSearch(grid_dict, 'gamma')
        print("The best gamma for the sigmoid kernel is %s" % best_gamma)
        svm = SVC(kernel='sigmoid', gamma=best_gamma)
        svm.fit(self.x_train, self.y_train)
        print("The best testing error rate for the sigmoid kernel SVM is %.4f" % (1 - svm.score(self.x_test, self.y_test)))
        return error

def main():
    data = DataHelper.Data()
    x_train, y_train, x_test, y_test, _ = data.loadData("hw2_data_2.txt", 20, 700)

    # radial kernel
    svm_radial = SVM(x_train, y_train, x_test, y_test, kernel="RADIAL", gamma_range=np.logspace(-3, 2, 6))
    svm_radial.run()

    # sigmoid kernel
    svm_sigmoid = SVM(x_train, y_train, x_test, y_test, kernel="SIGMOID", gamma_range=np.logspace(-3, 2, 6))
    svm_sigmoid.run()

    # polynomial kernel
    svm_poly = SVM(x_train, y_train, x_test, y_test, kernel="POLYNOMIAL", degree_range=range(1, 11))
    svm_poly.run()

    plt.show()

if __name__ == '__main__':
    main()