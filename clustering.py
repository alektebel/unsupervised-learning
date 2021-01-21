import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randint

def n_colors(n):
    """Generates a list of n different color in hexadecimal notation"""
    arr = []
    for i in range(n):
        arr.append('#%06X' % randint(0, 0xFFFFFF))
    return arr

class k_means():
    """This class implements the k-means algorithm"""
    def __init__(self, k):
        self.k = k
    def clusters(self, X):
        #Find the nearest points to the lastest means
        k = np.argmin(X@self.means, axis = 1)
        return k

    def fit(self, X, epochs):
        #Extract the predicted labels of the examples in the data.
        (m, n) = X.shape
        index = []
        for i in range(self.k):
            index.append(np.random.randint(0,m))
        #Initialize from a random sample of k elements of the dataset the means
        self.means = X[index,:]

        for i in range(epochs):

            self.labels = self.clusters(X)
            for label in range(self.k):
                #Extract the examples labeled with the label label
                mask = (label == self.labels)
                l_X = X*np.expand_dims(mask, axis = 1)
                self.means[:,label] = np.sum(l_X, axis = 0)/l_X.shape[0]
        return self.labels
    
def main():
    df = pd.read_csv('./datasets/Mall_Customers.csv')
    df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    k = 3
    alg = k_means(3)
    X = df.to_numpy()
    #Standarize data
    X = (X-np.mean(X, axis = 0))/np.std(X, axis = 0)
    labels = alg.fit(X, epochs = 5)

    colors = n_colors(3)
    rgb = []
    for label in labels:
        rgb.append(colors[label])

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],color = rgb, cmap=plt.cm.nipy_spectral,
           edgecolor='k')
    plt.show()

if __name__ == '__main__':
    main()