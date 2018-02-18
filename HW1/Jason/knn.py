import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Knn(object):

	def __init__(self, X, y, K):
		self.X = X
		self.y = y
		self.K = K
		self.res = np.array([])

	def predict(self, x):
		res = []
		for p in x:
			distances = np.array([np.linalg.norm(i-p) for i in self.X])
			kIndex = distances.argsort()[:self.K]
			values = y[kIndex]
			res.append(0 if np.sum(values) <= values.size // 2 else 1)

		self.res = np.array(res)
		return np.array(res)

	def getSensitivity(self, y_test):
		return np.dot(self.res, y_test)/y_test.sum()

	def getFDR(self, y_test):
		tp = np.dot(self.res, y_test)
		fp = 0
		for i in range(y_test.size):
			fp += self.res[i] == 1 and y_test[i] == 0

		return fp*1.0/(tp + fp)

	def getSpecificity(self, y_test):
		count = 0
		for i in range(y_test.size):
			count += self.res[i] == 0 and y_test[i] == 0

		return count / list(y_test).count(0)

## helper method for plotting test data
def plotTestData(X_test, y_test, res):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	for index, (i,j) in enumerate(X_test):
		color = 'blue' if y_test[index] == res[index] else 'red'
		marker = 'x' if y_test[index] else 'o'
		ax.scatter(i,j, c=color, marker=marker)


## Load Data
df = pd.read_csv('HW_1_training.txt', sep='\t')
X = np.array(df.loc[:, ['x1', 'x2']])
y = np.array(df['y'])

df = pd.read_csv('HW_1_testing.txt', sep='\t')
X_test = np.array(df.loc[:, ['x1', 'x2']])
y_test = np.array(df['y'])

for k in (1,5,10):
	## Initialize Classifier
	clf = Knn(X, y, k)
	res = clf.predict(X_test)

	## Plot Data
	plotTestData(X_test, y_test, res)
	plt.title("K Nearest Neighbour (k = %s)" %k)

	## Calculate different rate
	print("When k = %s" %k)
	print("The sensitivity rate is %.2f"  % clf.getSensitivity(y_test))
	print("The specificity rate is %.2f"  % clf.getSpecificity(y_test))
	print("The false discovery rate rate is %.2f"  % clf.getFDR(y_test))
	print("")

plt.show()