#Felix Song, 11/8
#HW6

import numpy as np
import matplotlib.pyplot as plt

# Load the vector x and basis U from CSV files
x = np.loadtxt('x.csv', delimiter=',')
U = np.loadtxt('U.csv', delimiter=',')

e1 = np.zeros(1000)
e1[0] = 1

encoded_e1 = np.dot(U.T,e1)

plt.figure(figsize=(10, 6))
plt.stem(encoded_e1, markerfmt='ro', use_line_collection=True)
plt.title('Representation of e1 in the Basis U')
plt.xlabel('Basis Index')
plt.ylabel('Coefficient Value')
plt.show()

#%%

#1b

c = np.dot(U.T,x)

plt.figure(figsize=(10, 6))
plt.stem(c, markerfmt='ro', use_line_collection=True)
plt.title('Representation of e1 in the Basis U')
plt.xlabel('Basis Index')
plt.ylabel('Coefficient Value')
plt.show()

#%%

#1c

m = 60
n = 1000
A = (1 / np.sqrt(m)) * np.random.randn(m, n)
b = np.dot(A, c)



#%%

#1d

def IHT(A, b, k):
    # Get the size of the matrix A
    m, n = A.shape

    # Initialization
    y = np.zeros((n, 1))

    for _ in range(10):
        y = y - A.T @ (A @ y - b)
        idx = np.argsort(np.abs(y.flatten()))[::-1]
        z = np.zeros((n, 1))
        z[idx[:k]] = y[idx[:k]]
        y = z

    return y

k = 3
IHT(A,b,k)
#%%

#2a
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import spdiags

X = np.loadtxt('Moon.csv', delimiter=',')
labels = np.loadtxt('Moon_Label.csv', delimiter=',')

#We can start with a initial value of sigma being 1
sigma = 1
distances_squared = cdist(X, X, 'sqeuclidean')
W = np.exp(-distances_squared / (2 * sigma**2))
D = np.diag(np.sum(W, axis=1))
L = D - W

plt.figure(figsize=(8, 8))
plt.imshow(L, cmap='viridis', interpolation='nearest')
plt.title('Laplacian Matrix')
plt.colorbar()
plt.show()

#%%

#2b
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding

eigenvalues, eigenvectors = np.linalg.eigh(L)

# Sort eigenvalues and corresponding eigenvectors
sorted_indices = np.argsort(eigenvalues)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

embedding_result = eigenvectors[:, 1:3]

kmeans = KMeans(n_clusters=k, random_state=0)
labels1 = kmeans.fit_predict(embedding_result[:, 1:3])

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels1, cmap='bwr', edgecolors='k', s=50)
plt.title('K-means Clustering on Spectral Embedding')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#%%

plt.figure(figsize=(8, 6))
plt.scatter(embedding_result[:, 0], embedding_result[:, 1], c=labels1, cmap='bwr', edgecolors='k', s=50)
plt.title('Spectral Embedding')
plt.xlabel('2nd Eigenvector')
plt.ylabel('3rd Eigenvector')
plt.show()


#%%

#the column way
def calc1(A,x):
    num = np.zeros(A.shape[0])
    for a in range(A.shape[1]):
        num += x[a] * A[:,a]
    return num

#the row way
def calc2(A,x):
    num = np.zeros(A.shape[0])
    for b in range(A.shape[0]):
        num[b] = A.T[:,b] @ x
    return num

#%%
import time
n = 20000
iterations = 30

column_time = np.zeros(iterations)
row_time = np.zeros(iterations)


for c in range(iterations):
    A = np.random.rand(n, n)
    x = np.random.rand(n)
    
    startTime = time.time()
    calc1(A,x)
    column_time[c] = time.time()-startTime
    
    startTime = time.time()
    calc2(A,x)
    row_time[c] = time.time()-startTime
    
avg_time_rows = np.mean(column_time)
avg_time_columns = np.mean(row_time)











