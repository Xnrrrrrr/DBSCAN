import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the clustered data
data = pd.read_csv('C:/Users/Bank/Desktop/Clustered_TrafficData.csv')

# 1. Examine the first few rows to confirm the data structure
print("First few rows of the dataset:")
print(data.head())

# Convert non-numeric columns to numeric (if applicable)
# Identifying non-numeric columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
print(f"\nNon-numeric columns identified: {non_numeric_columns}")

# Convert each non-numeric column to numeric (using LabelEncoder for categorical columns)
label_encoder = LabelEncoder()
for col in non_numeric_columns:
    data[col] = label_encoder.fit_transform(data[col])
    print(f"Converted column '{col}' to numeric values.")

# 2. Count the number of data points in each cluster (including noise)
cluster_counts = data['Cluster'].value_counts()
print("\nCluster distribution:")
print(cluster_counts)

# 3. Check for noise points (DBSCAN - usually labeled as -1)
noise_points = data[data['Cluster'] == -1]
print(f"\nNumber of noise points (Cluster -1): {len(noise_points)}")
print("First few noise points:")
print(noise_points.head())

# 4. Visualize the distribution of the clusters using a scatter plot
# For simplicity, using 'Time' and 'Length' for visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Time', y='Length', hue='Cluster', palette='Set1', legend='full', s=60)
plt.title('Clustered Traffic Data (Time vs Length)', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Length', fontsize=14)
plt.legend(title='Cluster')
plt.show()

# 5. Pairplot to visualize relationships between multiple features and clusters
sns.pairplot(data, hue='Cluster', palette='Set1', diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50})
plt.suptitle('Pairplot of Clustering Results (Time, Length, etc.)', fontsize=16)
plt.show()

# 6. Cluster statistics: Calculate the mean of all features for each cluster
print("\nCluster Statistics (Mean of Features for Each Cluster):")
cluster_stats = data.groupby('Cluster').mean()
print(cluster_stats)

# 7. Cluster Min/Max/Std: Statistical summary for each cluster
cluster_min_max = data.groupby('Cluster').agg(['min', 'max', 'std'])
print("\nCluster Statistics (Min, Max, Std):")
print(cluster_min_max)

# 8. Visualizing the distribution of features within each cluster
# Boxplots for 'Time' and 'Length' by cluster
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='Time', data=data, palette='Set1')
plt.title('Boxplot of Time Feature by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Time', fontsize=14)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='Length', data=data, palette='Set1')
plt.title('Boxplot of Length Feature by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Length', fontsize=14)
plt.show()

# 9. Evaluate protocol distribution in each cluster (if applicable)
if 'Protocol' in data.columns:
    protocol_distribution = data.groupby('Cluster')['Protocol'].value_counts()
    print("\nProtocol Distribution by Cluster:")
    print(protocol_distribution)

# 10. Investigate any unusual traffic characteristics (e.g., outliers)
# Check if any clusters have particularly high or low values for 'Time' or 'Length'
outliers = data[(data['Time'] > data['Time'].quantile(0.95)) | (data['Length'] > data['Length'].quantile(0.95))]
print("\nPossible Outliers in Traffic Data (High Time or Length):")
print(outliers.head())

# 11. Plot histograms for each feature (Time, Length) for each cluster
plt.figure(figsize=(14, 6))
sns.histplot(data, x='Time', hue='Cluster', kde=True, multiple='stack', palette='Set1', bins=30)
plt.title('Histogram of Time by Cluster', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

plt.figure(figsize=(14, 6))
sns.histplot(data, x='Length', hue='Cluster', kde=True, multiple='stack', palette='Set1', bins=30)
plt.title('Histogram of Length by Cluster', fontsize=16)
plt.xlabel('Length', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

# 12. Correlation Matrix of features (only numeric)
correlation_matrix = data.corr()
print("\nCorrelation Matrix of Features:")
print(correlation_matrix)

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Traffic Features', fontsize=16)
plt.show()

# 13. Inspect any additional statistics of interest (e.g., specific packet sizes or durations)
# You can filter and analyze specific data points or clusters that seem unusual or significant
specific_cluster = 1  # Example: Cluster 1
specific_cluster_data = data[data['Cluster'] == specific_cluster]
print(f"\nData for Cluster {specific_cluster}:")
print(specific_cluster_data.head())

# Save cluster statistics to a CSV file for further analysis
cluster_stats.to_csv('Clustered_Statistics.csv')
