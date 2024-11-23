import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the clustered data
data = pd.read_csv('C:/Users/Bank/Desktop/Clustered_TrafficData1.csv')

# 1. Initial Examination
print("----- INITIAL DATA EXAMINATION -----")
print("First few rows of the dataset:")
print(data.head())

# Identify non-numeric columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_columns) > 0:
    print(f"Non-numeric columns identified: {list(non_numeric_columns)}")
else:
    print("No non-numeric columns found.")

# Convert non-numeric columns to numeric (including 'Protocol')
label_encoder = LabelEncoder()
for col in non_numeric_columns:
    data[col] = label_encoder.fit_transform(data[col])
    print(f"Converted column '{col}' to numeric values.")

print(f"\nDataset now contains {data.shape[0]} rows and {data.shape[1]} columns.")
print("----- END OF INITIAL EXAMINATION -----\n")

# 2. Cluster Distribution
print("----- CLUSTER DISTRIBUTION -----")
cluster_counts = data['Cluster'].value_counts()
print("Number of data points in each cluster:")
print(cluster_counts)

# Highlight largest and smallest clusters
largest_cluster = cluster_counts.idxmax()
smallest_cluster = cluster_counts.idxmin()
print(f"\nLargest cluster is Cluster {largest_cluster} with {cluster_counts[largest_cluster]} data points.")
print(f"Smallest cluster (excluding noise) is Cluster {smallest_cluster} with {cluster_counts[smallest_cluster]} data points.")

# Noise Points
noise_points = data[data['Cluster'] == -1]
print(f"\nNumber of noise points (Cluster -1): {len(noise_points)}")
if len(noise_points) > 0:
    print("First few noise points:")
    print(noise_points.head())
else:
    print("No noise points detected.")
print("----- END OF CLUSTER DISTRIBUTION -----\n")

# 3. Visualizations
print("----- VISUALIZATIONS -----")
print("Generating scatter plot of Time vs Length...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Time', y='Length', hue='Cluster', palette='Set1', legend='full', s=60)
plt.title('Clustered Traffic Data (Time vs Length)', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Length', fontsize=14)
plt.legend(title='Cluster')
plt.show()

print("Generating pairplot for feature relationships...")
sns.pairplot(data, hue='Cluster', palette='Set1', diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50})
plt.suptitle('Pairplot of Clustering Results (Time, Length, etc.)', fontsize=16)
plt.show()
print("----- END OF VISUALIZATIONS -----\n")

# 4. Cluster Statistics
print("----- CLUSTER STATISTICS -----")
print("Calculating mean, min, max, and standard deviation of features for each cluster...")
cluster_stats = data.groupby('Cluster').mean()
print("\nMean values for each cluster:")
print(cluster_stats)

cluster_min_max = data.groupby('Cluster').agg(['min', 'max', 'std'])
print("\nMin, Max, and Standard Deviation of features for each cluster:")
print(cluster_min_max)

# show interesting insights
print("\nInsights:")
print("- Check for clusters with extreme values in 'Time' or 'Length' (e.g., Cluster 0 and Cluster 1).")
print("- Small standard deviations might indicate tighter grouping of values.")
print("----- END OF CLUSTER STATISTICS -----\n")

# 5. Boxplots for Features by Cluster
print("----- FEATURE DISTRIBUTION -----")
print("Generating boxplots for feature distributions by cluster...")
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
print("----- END OF FEATURE DISTRIBUTION -----\n")

# 6. Protocol Analysis by Cluster
print("----- PROTOCOL ANALYSIS -----")
if 'Protocol' in data.columns:
    protocol_distribution = data.groupby('Cluster')['Protocol'].value_counts()
    print("Protocol Distribution by Cluster:")
    print(protocol_distribution)

    print("Generating bar plot of Protocol distribution across clusters...")
    protocol_cluster_df = data.groupby(['Cluster', 'Protocol']).size().reset_index(name='Count')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=protocol_cluster_df, x='Cluster', y='Count', hue='Protocol', palette='Set2')
    plt.title('Protocol Distribution Across Clusters', fontsize=16)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(title='Protocol')
    plt.show()
    print("- Protocols vary significantly across clusters; focus on clusters with dominant Protocols for deeper analysis.")
else:
    print("No 'Protocol' column detected in the dataset.")
print("----- END OF PROTOCOL ANALYSIS -----\n")

# 7. Outlier Detection
print("----- OUTLIER DETECTION -----")
outliers = data[(data['Time'] > data['Time'].quantile(0.95)) | (data['Length'] > data['Length'].quantile(0.95))]
print(f"Found {len(outliers)} potential outliers in the dataset based on 'Time' and 'Length' thresholds.")
if len(outliers) > 0:
    print("First few outliers detected:")
    print(outliers.head())
else:
    print("No significant outliers detected.")
print("----- END OF OUTLIER DETECTION -----\n")

# Histograms for Features
print("Generating histograms for feature distributions by cluster...")
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

# 8. Correlation Matrix
print("----- CORRELATION ANALYSIS -----")
correlation_matrix = data.corr()
print("Correlation Matrix of Features:")
print(correlation_matrix)

print("Generating correlation heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Traffic Features', fontsize=16)
plt.show()

print("- Look for highly correlated features that might indicate redundant data or strong patterns.")
print("----- END OF CORRELATION ANALYSIS -----\n")

# 9. Specific Cluster Analysis
specific_cluster = 1  # Example: Cluster 1
specific_cluster_data = data[data['Cluster'] == specific_cluster]
print(f"----- DETAILED ANALYSIS FOR CLUSTER {specific_cluster} -----")
print(f"First few rows of data for Cluster {specific_cluster}:")
print(specific_cluster_data.head())
print(f"Cluster {specific_cluster} contains {len(specific_cluster_data)} data points.")
print("----- END OF CLUSTER DETAIL -----\n")

# 10. Save Cluster Statistics
cluster_stats.to_csv('Clustered_Statistics1.csv')
print("Cluster statistics saved to 'Clustered_Statistics1.csv'.")
print("----- SCRIPT COMPLETED -----")
