#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Author: Zhu
E-mail: gs.ywzhu19@gzu.edu.cn
Data set:
Constructed two types of imbalanced data with an imbalance ratio of 0.5
Number of minority class samples: 10
Number of minority class samples: 20
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.mixture import GaussianMixture


"""
Inputs
-----
X: sample features of the input data, minority class samples in the front, majority class samples in the back
y: Labels, with minority example labeled as 1,majority  example labeled as 0
beta: Degree of imbalance desired. A 1 means the positive and negative examples are perfectly balanced.
K: The value of K for the KNN algorithm, meaning the number of neighbors to find
threshold:  The maximum tolerated degree of class imbalance ratio
N: Judgment factor to determine if a minority class sample is radical,  0 < N <= K

Variables
-----
xi: Minority example
xzi: A minority example inside the neighbourhood of xi,when xi is not an aggressive minority class sample.
xci：The center of a sample cluster closest to xi
si: A new sample of synthesis
ms: Amount of data in minority class
ml: Amount of data in majority class
clf: k-NN classifier model
gmm: GMM clustering algorithm based on EM algorithm
x_mean: Clustering center of minority class samples
d: The imbalance ratio of the data set 
beta: Degree of imbalance desired
G: Amount of data to generate
ND_Majority: Neighborhood density of the majority class sample
Di: Sample neighborhood density in different cases
Larger Di means more sample size needs to be synthesized.
In addition, Di allows the user to run CASS with a small value of K
Minority_per_xi: All the minority data's index by neighbourhood
Nor_Di: Normalized Di, where sum = 1
Gi: Amount of data to generate per neighbourhood (indexed by neighbourhoods corresponding to xi)

Returns
-----
syn_data: New synthetic minority data created
"""


def CASS(X, y, beta, K, threshold, n, N):
    ms = int(sum(y))
    ml = len(y) - ms
    clf = neighbors.KNeighborsClassifier(weights='distance', algorithm='auto')
    clf.fit(X, y)

    # Using GMM to calculate the cluster centers of minority classes
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0)
    x = X[0:ms,:]
    gmm.fit(x)
    x_mean = gmm.means_

    # Calculate the imbalance of the data set
    d = np.divide(ms, ml)
    if d > threshold:
        return print(u"The data set is not imbalanced enough!!!")

    # Calculate the number of samples to be synthesized
    G = (ml - ms) * beta

    # Judgment and evaluation of different samples
    Di = []
    Minority_per_xi = []
    for i in range(ms):
        xi = X[i, :].reshape(1, -1)
        neighbours = clf.kneighbors(xi, n_neighbors=K,return_distance=False)[0]
        neighbours = neighbours[1:]
        minority = []
        for value in neighbours:
            if value < ms:
                minority.append(value)
        Minority_per_xi.append(minority)

        # Calculate the Di
        count1 = np.sum(neighbours < ms)
        count2 = np.sum(neighbours >= ms)
        if count1 != 0 and count2 != 0:
            nd_i = count1 / (K-1)
        else:
            # So that smaller K values are acceptable in CASS
            nd_i = count1 / (2*K)
        Di.append(nd_i)
        # Normalize Di
        Nor_Di = []
        for nd_i in Di:
            Nor_nd_i = nd_i / sum(Di)
            Nor_Di.append(Nor_nd_i)
    # Calculate the number of synthetic data examples that will be generated for each minority example
    Gi = []
    for Nor_nd_i in Nor_Di:
        gi = round(Nor_nd_i * G)
        Gi.append(int(gi))

    # Generate synthetic examples
    syn_data = []
    for i in range(ms):
        xi = X[i, :].reshape(1, -1)
        Minority_number = len(Minority_per_xi[i])
        for j in range(Gi[i]):
            # Conservative minority class sample
            if len(Minority_per_xi[i]) >= (K / N):
                index = np.random.choice(Minority_per_xi[i])
                xzi = X[index, :].reshape(1, -1)
                # Synthesis method of new samples
                si = xi + (xzi - xi) * np.random.uniform(0, 1)
            # Radical minority class sample
            else:
                ND_Majority = (K-1 - Minority_number) / K
                dis = np.sqrt(np.sum(np.asarray(xi - x_mean) ** 2, axis=1))
                index = np.argmin(dis)
                xci = x_mean[index, :].reshape(1, -1)
                # Synthesis method of new samples
                si = xi + (xci - xi) * np.random.uniform(ND_Majority, 1)
            syn_data.append(si)

    # Build the data matrix
    data= []
    for values in syn_data:
        data.append(values[0])
    print("{} amount of minority class samples generated".format(len(data)))
    # Concatenate the positive labels with the newly made data
    labels = np.ones([len(data), 1])
    data_new = np.concatenate([labels, data], axis=1)
    # x_mean is used to draw the graph
    return data_new, x_mean


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # Features of the samples
    X = np.array(
        [[2.0, 2.0], [2.5, 3.5], [3.0, 2.0], [4.5, 2.5], [5.5, 3.0], [6.5, 3.0], [9.5, 3.0], [10.5, 2.5], [12.0, 2.0],
         [12.5, 3.5],
         [5.0, 4.0], [5.0, 3.0], [6.0, 5.0], [6.0, 4.0], [6.0, 3.0], [6.0, 2.0], [7.0, 5.0], [7.0, 4.0], [7.0, 3.0],
         [7.0, 2.0],
         [8.0, 5.0], [8.0, 4.0], [8.0, 3.0], [8.0, 2.0], [9.0, 5.0], [9.0, 4.0], [9.0, 3.0], [9.0, 2.0], [10.0, 4.0],
         [10.0, 3.0]])
    # Labeling of samples
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    #x_mean are used to represent the visualization of the data
    data_new, x_mean = CASS(X, y, beta=1, K=6, threshold=1, n=2, N=3)

    # Visualization of data
    old_x1_s = X[0:10,0]
    old_x2_s = X[0:10,1]
    new_x1 = data_new[:,1]
    new_x2 = data_new[:,2]
    aver_x1 = x_mean[:,0]
    aver_x2 = x_mean[:,1]
    old_x1_l = X[10:30, 0]
    old_x1_2 = X[10:30, 1]
    plt.scatter(old_x1_s, old_x2_s,  edgecolors='b', marker='o',  s=120,norm=0.95, alpha=1,
                label=u'Minority')
    plt.scatter(old_x1_l, old_x1_2, edgecolors='g', marker='s',  s=120,  norm=0.95, alpha=1,
                label=u'Majority')
    plt.scatter(new_x1, new_x2, edgecolors='gray', marker='o', s=120,   norm=0.95, alpha=0.8,
                label=u'Synthesized ')
    plt.scatter(aver_x1, aver_x2, edgecolors='purple', marker='*', s=120, norm=0.95, alpha=0.65,
                label=u'Cluster Center')
    plt.autoscale()
    plt.title(u'CASS', fontsize=20)
    # plt.grid(True)
    plt.legend(fancybox=True, framealpha=1,fontsize=12, loc='upper left')
    plt.show()

