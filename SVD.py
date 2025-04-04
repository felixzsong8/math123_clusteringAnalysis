import numpy as np
import matplotlib.pyplot as plt

n = 9
m = 10
size = 10000


data = np.zeros(size)
sigmaList = np.zeros(size)
uList = np.zeros((m,size))
vList = np.zeros((n, size))

for i in range(size):
    rng = np.random.default_rng()
    v = rng.standard_normal(n)
    u = rng.standard_normal(m)
    
    A = rng.standard_normal((n,m))
    
    U, Sigma, Vt = np.linalg.svd(A)
    
    sigmaList[i] = Sigma[0]
    uList[:, i] = U[:, 0]
    vList[:, i] = Vt[0, :]
    
    val = np.linalg.norm(np.dot(A, vList[:, i]) - (sigmaList[i] * uList[:, i]))
    data[i] = val
    
    
sum1 = 0.0
for j in range(size):
    sum1 += data[j]
    
sum1 /= 10000
    
#%%

import numpy as np
import matplotlib.pyplot as plt

n = 9
m = 10
size = 10000



sigma = 0.0
uConst = np.zeros(m)
vConst = np.zeros(n)
A = np.random.normal(size=(m,n))
for i in range(size):
    v = np.random.normal(size=n)
    u = np.random.normal(size=m)
    
    v = v/(np.linalg.norm(v))
    u = u/(np.linalg.norm(u))
    
    s = np.dot(u,(A @ v))
    if(s > sigma):
        sigma = s
        uConst = u
        vConst = v

val = np.linalg.norm((A @ vConst) - (sigma * uConst))
    
    
#%%
import numpy as np
A = np.zeros((3,4))
A[0][0] = 1
A[1][1] = 20
A[2][0] = 1
A[2][3] = 7

U, Sigma, V = np.linalg.svd(A)
    
#%%

#HW 3 
#Mathamatical aspects of Data Analysis
#Felix Song

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
 
 
# load the image and convert into
# numpy array
img = Image.open('flowers.png')
data = asarray(img)[:, :, 0]

#  shape
print(data.shape)

U, Sigma, V = np.linalg.svd(data)
    
    
#%%

#Problem d)
#Truncated matrix
s = np.zeros((602,602))
np.fill_diagonal(s, Sigma)
k = 100

Ahat = U[:, 0:k] @ s[0:k, 0:k] @ V[0:k, :]

#%%

image = Image.fromarray(Ahat)
image.show()

#%%
num = 0
for a in range(600):
    print(a)
    if(Sigma[a] <= Sigma[a+1]*1.01):
        num = a
        break


#%%

#HW 4
#Problem 6a
import numpy as np

def do_PCA(data):
    # Center the data 
    dataC = data - np.mean(data, axis=0)
    # Calculate the covariance matrix
    cov = (1/dataC.shape[0])*(np.transpose(dataC) @ dataC)
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    #sort
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # The principal components are the eigenvectors
    principal_components = eigenvalues

    return principal_components

#%%
#Problem 6b

import csv

with open('/Users/felixsong/Desktop/coding-tufts/DataA/gaussian_noisy.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = []
    for b in csv_reader:
        data.append([float(x) for x in b])
                
gaus_comp = do_PCA(data)

print(gaus_comp)
with open('/Users/felixsong/Desktop/coding-tufts/DataA/uniform_noisy.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data2 = []
    for b in csv_reader:
        data2.append([float(x) for x in b])
        
uni_comp = do_PCA(data2)
print(uni_comp)
                
             
#%%     
#Question 7a 
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

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colormap)
plt.title('MNIST Data - First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=['0', '6', '7'])
plt.colorbar()
plt.show()
                
#%%
#Problem 7bu
#Set an array for the number of principle components
k_array = [1]
for c in range(1,30):
    k_array.append(c*10)
    
#%%
value_array = []
for a in k_array:
    Error = 0
    count = 0
    pca = PCA(n_components=a)
    X_pca = pca.fit_transform(X)
    X_rec = pca.inverse_transform(X_pca)
    for b in range(data3.shape[0]):
        if(colormap[0][b] == 7):
            value = (np.linalg.norm(X[b] - X_rec[b])) ** 2
            Error += value
            count += 1
    print(count)
    value_array.append((1/count)*Error)
#%%
plt.plot(k_array, value_array)
plt.title("Average reconstruction error as number of components increases")
plt.xlabel("Number of principle components")
plt.ylabel("Error")
plt.show()

                
                
                

                
                
                
                
                
                
                
                










    
    