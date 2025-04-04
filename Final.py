import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time


def run_spectral_clustering(n, use_nystroem=False):
   data_subset = data.head(n)


   features = data_subset[['distance_traveled', 'total_fare']]
   scaler = StandardScaler()
   features_scaled = scaler.fit_transform(features)


   if use_nystroem:
       # Nyström approximation
       nystroem = Nystroem(kernel='rbf', n_components=100, random_state=42)
       features_nystroem = nystroem.fit_transform(features_scaled)
   else:
       features_nystroem = features_scaled


   num_clusters = 3
   start_time = time.time()
   spectral_clustering = SpectralClustering(n_clusters=num_clusters)
   data_subset['cluster'] = spectral_clustering.fit_predict(features_nystroem)
   end_time = time.time()


   elapsed_time = end_time - start_time
   silhouette_avg = silhouette_score(features_nystroem, data_subset['cluster'])


   return elapsed_time, silhouette_avg


# Load data
file_path = "train.csv"
data = pd.read_csv(file_path)


# Values of n to test
n_values = [5000, 100000, 200000]


# Run the program for different values of n with and without Nyström approximation
execution_times_normal = []
silhouette_scores_normal = []
execution_times_nystroem = []
silhouette_scores_nystroem = []


for n in n_values:
   # Without Nyström
   time_taken_normal, silhouette_avg_normal = run_spectral_clustering(n, use_nystroem=False)
   execution_times_normal.append(time_taken_normal)
   silhouette_scores_normal.append(silhouette_avg_normal)


   print(f"Execution time without Nyström for n={n}: {time_taken_normal:.4f} seconds")
   print(f"Silhouette Score without Nyström for n={n}: {silhouette_avg_normal:.4f}")


   # With Nyström
   time_taken_nystroem, silhouette_avg_nystroem = run_spectral_clustering(n, use_nystroem=True)
   execution_times_nystroem.append(time_taken_nystroem)
   silhouette_scores_nystroem.append(silhouette_avg_nystroem)


   print(f"Execution time with Nyström for n={n}: {time_taken_nystroem:.4f} seconds")
   print(f"Silhouette Score with Nyström for n={n}: {silhouette_avg_nystroem:.4f}")


# Plot the execution times
plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(n_values, execution_times_normal, marker='o', label='Without Nyström')
plt.plot(n_values, execution_times_nystroem, marker='o', label='With Nyström')
plt.title('Execution Time vs. Number of Rows (n)')
plt.xlabel('Number of Rows (n)')
plt.ylabel('Execution Time (seconds)')
plt.legend()


# Plot the silhouette scores
plt.subplot(1, 2, 2)
plt.plot(n_values, silhouette_scores_normal, marker='o', label='Without Nyström')
plt.plot(n_values, silhouette_scores_nystroem, marker='o', label='With Nyström', color='orange')
plt.title('Silhouette Score vs. Number of Rows (n)')
plt.xlabel('Number of Rows (n)')
plt.ylabel('Silhouette Score')
plt.legend()


plt.tight_layout()
plt.show()


