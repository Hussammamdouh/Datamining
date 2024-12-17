# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
file_path = 'house_data.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Step 2: Extract relevant features
X = data[['price', 'condition']].values

# Step 3: Normalize the data (optional but recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Use the Elbow Method to determine the optimal number of clusters
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method results
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Step 5: Apply K-Means clustering with the optimal number of clusters
optimal_clusters = 3  # Replace with the number observed from the Elbow Method
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Step 6: Add cluster labels to the original dataset
data['Cluster'] = labels

# Step 7: Visualize the clusters
plt.figure(figsize=(8, 6))
for cluster in range(optimal_clusters):
    plt.scatter(X_scaled[labels == cluster, 0], X_scaled[labels == cluster, 1], label=f'Cluster {cluster}')

# Plot the cluster centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='X', label='Centroids')

plt.title('K-Means Clustering of Annual Price and condition')
plt.xlabel('Price (Standardized)')
plt.ylabel('condition (Standardized)')
plt.legend()
plt.grid()
plt.show()

# Step 8: Save the dataset with cluster labels
data.to_csv('clustered_house_data.csv', index=False)
