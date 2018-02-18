import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from data_loader import DataLoader

#load data
data = DataLoader()
data.load()

trn_0 = data.trn_0
trn_1 = data.trn_1
tst_0 = data.tst_0
tst_1 = data.tst_1
trn_all = data.trn_all
tst_all = data.tst_all

#set prior here
equal_prior = False
prior_0 = None
prior_1 = None

if equal_prior:
    print('jack')
    prior_0 = 0.5
    prior_1 = 0.5
else:
    prior_0 = float(trn_0.shape[1]) / (trn_0.shape[1] + trn_1.shape[1])
    prior_1 = 1 - prior_0

#get the range in the training data
max_0 = trn_all[0].max()
min_0 = trn_all[0].min()
max_1 = trn_all[1].max()
min_1 = trn_all[1].min()

#calculating mean vector and covariance matrix based on two different class
mean_vector_0 = np.mean(trn_0[0:2], axis=1)
mean_vector_1 = np.mean(trn_1[0:2], axis=1)
cov_0 = np.cov(trn_0[0:2])
cov_1 = np.cov(trn_1[0:2])

num_feature = trn_0.shape[1] - 1

print("mean vector for class 0: ", mean_vector_0)
print("mean vector for class 1: ", mean_vector_1)

print("cov for class 0: ", cov_0)
print("cov for class 1: ", cov_1)

#Calculate the classification error rate
error_num = 0
for i in range(len(tst_all[0])):
    prob_1 = prior_0 * multivariate_normal.pdf((tst_all[0][i], tst_all[1][i]), mean_vector_0, cov_0)
    prob_2 = prior_1 * multivariate_normal.pdf((tst_all[0][i], tst_all[1][i]), mean_vector_1, cov_1)
    if prob_1 > prob_2:
        res = 0
    else:
        res = 1
    if res != tst_all[2][i]:
        error_num += 1

error_rate = float(error_num) / (len(tst_all[0]))
print("Classification error rate is: %f" % error_rate)

step = 0.02
x1 = np.arange(min_0, max_0 + step, step)
x2 = np.arange(min_1, max_1 + step, step)
x1_x2 = []
classes = []

#calculate the decision boundary using probability density function
print("Class 0 prior is ", prior_0)
print("Class 1 prior is ", prior_1)
for i in x1:
    for j in x2:
        prob_1 = prior_0 * multivariate_normal.pdf((i, j), mean_vector_0, cov_0)
        prob_2 = prior_1 * multivariate_normal.pdf((i, j), mean_vector_1, cov_1)
        x1_x2.append((i, j))
        if prob_1 > prob_2:
            classes.append(0)
        else:
            classes.append(1)

x1_x2 = np.array(x1_x2).T
# print(x1_x2.shape)
# print(len(classes))
color_map = ['red', 'blue']

#scatter plot the boundary
plt.figure(1)
fig = plt.scatter(x1_x2[0], x1_x2[1], c=classes, cmap=matplotlib.colors.ListedColormap(color_map))
plt.xlabel('x1')
plt.ylabel('x2')
# plt.scatter(tst_0[0], tst_0[1], c='g')
# plt.scatter(tst_1[0], tst_1[1], c='c')
plt.show()







