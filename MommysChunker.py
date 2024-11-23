# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:40:33 2024

@author: Bank
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN

# Define file path and columns to use
file_path = r'C:\Users\Bank\Desktop\TrafficData.csv'
usecols = ['Time', 'Source Port', 'Destnation Port', 'Length', 'Protocol']  # Include Protocol

# DBSCAN parameters
eps = 0.5  # distance threshold for clustering
min_samples = 10  # minimum number of samples for a point to be considered a core point

# Initialize the list to store chunks of clustered data
clustered_data = []

# Read and process the CSV file in chunks
chunksize = 100000  # Number of rows to read at a time
chunks = pd.read_csv(file_path, usecols=usecols, chunksize=chunksize)

# Initialize LabelEncoder for Protocol column
protocol_encoder = LabelEncoder()

# Process each chunk
for i, chunk in enumerate(chunks):
    try:
        # Drop rows with missing values
        chunk.dropna(inplace=True)

        # Encode the Protocol column (non-numeric categorical feature)
        chunk['Protocol'] = protocol_encoder.fit_transform(chunk['Protocol'])

        # Standardize the numeric columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(chunk[['Source Port', 'Destnation Port', 'Length', 'Protocol']])

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        chunk['Cluster'] = dbscan.fit_predict(scaled_data)

        # Append the chunk with clusters (including noise, label -1)
        clustered_data.append(chunk)

        print(f"Processed chunk {i+1} with {chunk['Cluster'].nunique()} unique clusters (including noise).")

    except Exception as e:
        print(f"Error processing chunk {i+1}: {e}")

# Combine all chunks into a final DataFrame
if clustered_data:
    final_clustered_data = pd.concat(clustered_data, ignore_index=True)
    print("Concatenation successful. Final data has shape:", final_clustered_data.shape)
else:
    print("No data processed.")

# Save the final clustered data to a new CSV file
output_file_path = r'C:\Users\Bank\Desktop\Clustered_TrafficData1.csv'
final_clustered_data.to_csv(output_file_path, index=False)
print(f"Clustered data saved to {output_file_path}")
