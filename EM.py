import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randint
from scipy.stats import multivariate_normal as MN

def n_colors(n):
    """Generates a list of n different color in hexadecimal notation"""
    arr = []
    for i in range(n):
        arr.append('#%06X' % randint(0, 0xFFFFFF))
    return arr
def get_variance(X):
    """This function returns the covariance matrix, from a dataset X"""
    (m, n ) = X.shape
    var = np.empty((X.shape[1], X.shape[1]))
    for i in range(m):
        var = var + X[i,:].T @ X[i,:]
    var = np.cov(X.T)
    return var
def get_mean(X):
    """Computes the mean of X, given that:
        -X.shape = (m, n)
        -m is the number of instances of the dataset.    
            """
    (m, n) = X.shape
    return np.sum(X, axis = 0)/m

def gaussian(x, mean, cov):
    """Compute the multidimensional gaussian function for a single example"""
    if np.linalg.det(cov) == 0:
        print('la matriz es singular.')
        print(cov)
    return MN.pdf(x, mean=mean, cov=cov)

def likely_labels(X, means, covs):
    """This function returns a vector with the most likely label of the distribution:
    -X dataset of examples

    -mean: matrix with the means of the distributions stacked horizontally (n, k). Let us remember that

    -var: tensor of shape (n, n, k) with the matrix variance across the third axis"""
    (m, n) = X.shape
    (n, k) = means.shape
    labels = np.zeros((m,))
    buffer = np.zeros((k,))
    for j in range(m):
        for i in range(k):
            buffer[i] = gaussian(X[j,:], means[:,i], covs[:,:,i])
        labels[j] = np.argmax(buffer)
    return labels


class EM():
    """Implement Expectation maximization algorithm"""
    def __init__(self, k):
        self.k = k
    def fit(self, X, epochs):
        #Extract the predicted labels of the examples in the data.
        (m, n) = X.shape
        #n is the number of features and m is the number of samples
        index = []
        for i in range(self.k):
            index.append(np.random.randint(0,m))

        #Initialize from a random sample of k elements of the dataset the means
        self.means = np.asarray(X[index,:]).T

        self.variances = np.random.randn(n,n,self.k)
        #Initialize the variances as identity matrices
        for j in range(self.k):
            self.variances[:,:, j] = np.eye(n)

        #Initialize the vector of labels
        labels = np.zeros(m,)

        for _ in range(epochs):
            labels = likely_labels(X, self.means, self.variances)
            for label in range(self.k):
                current_examples_labeled_as_label = X[labels == label]
                self.means[:, label] = get_mean(current_examples_labeled_as_label)
                self.variances[:,:,label] = get_variance(current_examples_labeled_as_label)
        return labels       

def main():
    df = pd.read_csv('./datasets/Mall_Customers.csv')
    df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    k = 3
    alg = EM(k)

    X = df.to_numpy()
    #Standarize data
    
    X = (X-np.mean(X, axis = 0))/np.std(X, axis = 0)

    labels = alg.fit(X, epochs = 5)

    colors = n_colors(k)
    rgb = []
    for label in labels:
        rgb.append(colors[int(label.tolist())])

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X[:, 0], X[:, 1],X[:,2],color = rgb, cmap=plt.cm.nipy_spectral,
           edgecolor='k')
    plt.show()
    
if __name__ == '__main__':
    main()