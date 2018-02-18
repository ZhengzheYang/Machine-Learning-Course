import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

## Load Training Data and Test data
df = pd.read_csv('HW_1_training.txt', sep='\t')
X = np.array(df.loc[:, ['x1', 'x2']])
y = np.array(df['y'])

df = pd.read_csv('HW_1_testing.txt', sep='\t')
X_test = np.array(df.loc[:, ['x1', 'x2']])
y_test = np.array(df['y'])

colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

def plotpdf(bw, kernels):
	'''
	Plotting pdf for each class estimated by Gaussian KDE
	'''
	for i in range(2):
		x1, x2 = X[y==i, 0], X[y==i, 1]
		values = np.vstack([x1,x2])
                print values
		kernels.append(gaussian_kde(values, bw_method=bw))

		xmin, xmax = x1.min(), x1.max()
		ymin, ymax = x2.min(), x2.max()

		A, B = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
		positions = np.vstack([A.ravel(), B.ravel()])
		kernel = kernels[i]
		ax.plot(positions[0], positions[1], kernel(positions),c=colors[i], alpha=0.5)

def plotTestPoints(kernels):
	## Plot test points
	ax = fig.add_subplot(212)
	k1, k2 = kernels
	count = 0.0
	for index, i in enumerate(X_test):
		resClass = 0 if k1(i) > k2(i) else 1
		marker = 'o'
		color = 'blue'
		if resClass != y_test[index]:
			marker = 'x'
			color = 'red'
			count += 1

		ax.plot(i[0], i[1], c=color, marker=marker)

	return count

for bw in (0.1, 1, 10):
	fig = plt.figure()
	ax = fig.add_subplot(211, projection='3d')
	kernels = []

	plotpdf(bw, kernels)
	count = plotTestPoints(kernels)
	print("The classification error rate for bandwidth = %s is %.2f" %(bw, count/X_test.shape[0]))

plt.show()
