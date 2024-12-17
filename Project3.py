# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv('pricerunner_aggregate.csv')

# Strip any leading/trailing spaces from the column names
data.columns = data.columns.str.strip()

# Step 2: Extract relevant features (assuming 'Product ID', 'Merchant ID', 'Category ID' are numeric)
X = data[['Product ID', 'Merchant ID', 'Category ID']].values

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
optimal_clusters = 5  # Replace with the number observed from the Elbow Method
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

plt.title('K-Means Clustering of Products')
plt.xlabel('Product ID (Standardized)')
plt.ylabel('Merchant ID (Standardized)')
plt.legend()
plt.grid()
plt.show()