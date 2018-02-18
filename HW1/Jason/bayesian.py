import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import multivariate_normal

## A simple implementation of Bayesian Decision Boundary
class Bayesian(object):

	def __init__(self, priors=[0.5, 0.5]):
		self.priors = priors

	def fit(self, X, y):

		_classNum = np.unique(y)
		_groupX = []
		self.means = []
		self.covs = []

		for i in np.unique(y):
			_groupX.append(X[y == i])

		for group in _groupX:
			self.means.append(np.mean(group, axis = 0))
			self.covs.append(np.cov(group.T))

	def predict(self, X_test):
		_res = []
		for x in X_test:
			_probs = np.array([multivariate_normal.pdf(x, mean=self.means[i], cov=self.covs[i])*self.priors[i] 
							for i in range(len(self.means))])
			_res.append(np.argmax(_probs))

		return _res

	def getMeanVectors(self):
		return self.means 

	def getCovs(self):
		return self.covs

	def score(self, X_test, y_test):
		_res = self.predict(X_test)
		error = sum([i ^ j for i,j in zip(_res, y_test)])
		return error * 1.0 / len(y_test)


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
	                       np.arange(x2_min, x2_max, resolution))

	Z = np.array(classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T))
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], 
		        y=X[y == cl, 1],
		        alpha=0.8, 
		        c=colors[idx],
		        marker=markers[idx], 
		        label=cl, 
		        edgecolor='black')

	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend(loc='best')


## Load Training Data and Test data
df = pd.read_csv('HW_1_training.txt', sep='\t')
X = np.array(df.loc[:, ['x1', 'x2']])
y = np.array(df['y'])

df = pd.read_csv('HW_1_testing.txt', sep='\t')
X_test = np.array(df.loc[:, ['x1', 'x2']])
y_test = np.array(df['y'])


### Question 1:

clf = Bayesian()
clf.fit(X,y)
meanVectors, covMatrices = clf.getMeanVectors(), clf.getCovs()

for i in range(2):
	print("The mean vector for class %d is: %s" % (i, meanVectors[i]))
	print("The covariance matrix for class %d is: " % i)
	print(covMatrices[i])
	print("")

## Draw Decision Boundary without Prior
plt.figure(1)
plt.title('Naive Bayesian Boundary with no priors')
plot_decision_regions(X, y, classifier=clf)

## Calculate Test Error
print('The classification error rate for Bayesian Decision Boundary without priors is: %.2f'
				 %(clf.score(X_test,y_test)))


## Draw Decision Boundary with Prior

# Calculate priors
p0 = list(y).count(0)*1./y.size
priors = [p0, 1-p0]

plt.figure(2)
plt.title('Naive Bayesian Boundary with priors')
clf2 = Bayesian(priors=priors)
clf2.fit(X, y)

plot_decision_regions(X, y, classifier=clf2)
plt.show()

## Calculate Test Error

print('The classification error rate for Bayesian Decision Boundary with priors is: %.2f' 
				%(clf2.score(X_test,y_test)))





