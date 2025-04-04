#Felix Song
#HW5
#3a

import numpy as np

def kMeans(data, k, maxIterations):
    
    #create randomized clusters
    np.random.seed(0)
    cluster = data[np.random.choice(len(data), k, replace=False)]

    for _ in range(maxIterations):
        # Assign each data point to the nearest cluster
        distances = np.linalg.norm(data[:, np.newaxis] - cluster, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update cluster centroids
        new_cluster = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(cluster == new_cluster):
            break

        cluster = new_cluster

    return labels

#%%

#3bi

import csv

with open('/Users/felixsong/Desktop/coding-tufts/DataA/gaussians.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = []
    for b in csv_reader:
        data.append([float(x) for x in b[0:2]])
        
data1 = np.array(data)
clusterData = kMeans(data1, 2, 10)

#%%
import matplotlib.pyplot as plt

x = data1[:, 0]
y = data1[:, 1]

plt.scatter(x, y, c=clusterData)
plt.show()

#%%

#3bii

#Define a new function to get error
def kMeansErr(data, k, maxIterations):
    error = []
    #create randomized clusters
    np.random.seed(0)
    cluster = data[np.random.choice(len(data), k, replace=False)]

    for i in range(maxIterations):
        # Assign each data point to the nearest cluster
        distances = np.linalg.norm(data[:, np.newaxis] - cluster, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update cluster centroids
        new_cluster = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(cluster == new_cluster):
            break

        cluster = new_cluster
        err = np.sum(np.min(distances, axis=1))
        error.append(err)
        
        if i > 0 and abs(error[i] - error[i - 1]) < 1e-4:
            break;
        
    return error

#%%

kError = kMeansErr(data1, 2, 10)
plt.plot(kError, marker='o', linestyle='-')
plt.show()


#%%
#3ci

import csv

with open('/Users/felixsong/Desktop/coding-tufts/DataA/ellipses.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = []
    for b in csv_reader:
        data.append([float(x) for x in b[0:2]])
        
data2 = np.array(data)
clusterData = kMeans(data2, 2, 10)

#%%
import matplotlib.pyplot as plt

x = data2[:, 0]
y = data2[:, 1]

plt.scatter(x, y, c=clusterData)
plt.show()


#%%

#3cii)


kError1 = kMeansErr(data2, 2, 20)
plt.plot(kError1, marker='o', linestyle='-')
plt.show()

#%%


#4a
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io

file = scipy.io.loadmat('mnist_067.mat')
data3 = file["data_067"]
colormap = file["label_067"]

# Get Data
X = data3[:, :-1]
y = data3[:, -1]

# Do PCA and project the data onto the first two principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

label = kMeans(X_pca, 3, 100)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, cmap='viridis', s=1)
plt.title('K-Means Clustering (K=3) of MNIST Data (0, 6, 7)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()







