# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 00:40:30 2025

@author: angel
"""

# -*- coding: utf-8 -*-


#%% 1. Imports and Setup
import pandas as pd
#library for data manipulation and analysis, providing data structures like DataFrames.
import numpy as np
#A library for numerical computing with support for arrays, matrices, and mathematical functions.
import matplotlib.pyplot as plt
#A plotting library for creating static, animated, and interactive visualisations.
import seaborn as sns
#A statistical data visualisation library built on top of Matplotlib for attractive and informative graphics.
from sklearn.preprocessing import StandardScaler
#A tool to standardise features by removing the mean and scaling to unit variance.
from sklearn.model_selection import train_test_split
#A tool to standardise features by removing the mean and scaling to unit variance.
from sklearn.ensemble import RandomForestRegressor
#ML model using multiple decision trees for regression tasks.
from sklearn.metrics import mean_squared_error, r2_score
#Evaluate model performance using error and correlation metrics.
from sklearn.cluster import KMeans
#Clustering algorithm that partitions data into k clusters.
from sklearn.decomposition import PCA
# reduce dimensionality by transforming data into principal components.
from sklearn.ensemble import IsolationForest
#anomaly detection model that isolates outliers using a tree-based approach.


#%% 2.Loading the data into the our data frame 
df = pd.read_csv(r"C:\Users\angel\OneDrive - University of Greenwich\Data Mining 1916\archive (1)\butter_e_energy2.csv")

#Creating a copy of our dataframe
df_copy = df.copy()

# Clean data
print("Missing values:\n", df_copy.isnull().sum())
#Check for missing values and duplicates
print("Duplicates:", df_copy.duplicated().sum())
#Drop any duplicate rows
df_copy = df_copy.drop_duplicates()
#%%
#To check the time frame the data is recorded through
print("Earliest date:", df['timestamp'].min())
print("Latest date:", df['timestamp'].max())
#Earliest date: 2022-09-22T16:28:01.000Z
#Latest date: 2022-09-30T09:30:01.000Z

#%% 3. Feature Engineering

# Convert the 'timestamp' column to pandas datetime format
df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

# Extract features from timestamp

# Extract hour from timestamp (0–23)
df_copy['hour'] = df_copy['timestamp'].dt.hour

# Extract day of the week (0 = Monday, 6 = Sunday)
df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
# Create binary weekend indicator (1 if Saturday or Sunday, else 0)
df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)  # Friday-Sunday=1
# Create categorical column for weekend/weekday
df_copy['day_type'] = df_copy['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
# Extract day of the month (1–31)
df_copy['day_of_month'] = df_copy['timestamp'].dt.day

# Node encoding (Target Encoding with smoothing)
# First - Calculate global mean of energy consumption (watts)
global_mean = df_copy['watts'].mean()

# Then - Calculate mean energy use per node
node_target_mean = df_copy.groupby('node')['watts'].mean()

# Step 3: Smoothing: Use a weighted average between the global mean and the node mean
# Set regularisation parameter for target encoding
alpha = 10  
# Count number of samples per node
node_counts = df_copy.groupby('node').size()
# Perform smoothed target encoding for 'node' using prior mean and node mean
node_target_encoded = (node_target_mean * node_counts + global_mean * alpha) / (node_counts + alpha)

# Map encoded values to each row based on node
df_copy['node_encoded'] = df_copy['node'].map(node_target_encoded)

# Drop original 'node' column after encoding
df_copy = df_copy.drop(columns=['node'])

#%% 4. Exploratory Data Analysis (EDA)
# 4.1 Temporal Trends
# Create a 2x2 subplot figure for multiple plots
fig, ax = plt.subplots(2, 2, figsize=(16, 12))

# Weekly Patterns
# Group data by day of week and type, then calculate mean energy use
weekly_avg = df_copy.groupby(['day_of_week', 'day_type'])['watts'].mean().unstack()


# Plot weekly energy averages as stacked bar chart
weekly_avg.plot(kind='bar', ax=ax[0,0], stacked=True, color=['blue', 'red'])
ax[0,0].set_title('Weekly Energy Use by Day Type')
ax[0,0].set_ylabel('Watts')

# Daily Trends
# Calculate average energy per day of the month
daily_avg = df_copy.groupby('day_of_month')['watts'].mean()
# Plot daily energy averages as a line chart
daily_avg.plot(ax=ax[0,1], marker='o', color='green')
# Draw horizontal line representing the monthly average
ax[0,1].axhline(daily_avg.mean(), color='r', linestyle='--', label='Monthly Avg')
ax[0,1].set_title('Daily Energy Use')
ax[0,1].legend()

# Hourly Heatmaps
# Filter weekday data (is_weekend = 0)
weekday_data = df_copy[df_copy['is_weekend'] == 0]
# Filter weekend data (is_weekend = 1)
weekend_data = df_copy[df_copy['is_weekend'] == 1]

# Create heatmap of hourly energy use for weekdays
sns.heatmap(weekday_data.pivot_table(index='hour', columns='day_of_month', values='watts'),
            ax=ax[1,0], cmap='YlOrRd', cbar_kws={'label': 'Watts'})
ax[1,0].set_title('Weekday Hourly Patterns')

# Create heatmap of hourly energy use for weekends
sns.heatmap(weekend_data.pivot_table(index='hour', columns='day_of_month', values='watts'),
            ax=ax[1,1], cmap='YlOrRd', cbar_kws={'label': 'Watts'})
ax[1,1].set_title('Weekend Hourly Patterns')

plt.tight_layout()
plt.show()
#%%EDA
# Identify top 5 nodes with the highest average watts
top_nodes = df_copy.groupby('node_encoded')['watts'].mean().nlargest(5).index  

# Filter dataset to include only the top nodes and sort by hour  
top_nodes_data = df_copy[df_copy['node_encoded'].isin(top_nodes)].sort_values('hour')  

# Set figure size for better visibility  
plt.figure(figsize=(12, 6))  

# Create line plot to show hourly energy usage trends for top nodes  
sns.lineplot(  
    x='hour',  # X-axis represents the hour of the day  
    y='watts',  # Y-axis represents the average watts  
    hue='node_encoded',  # Different colours for each node  
    data=top_nodes_data,  # Filtered dataset with top nodes  
    estimator='mean',  # Aggregate data by mean value  
    errorbar=None,  # Disable error bars for cleaner visualisation  
    linewidth=2,  # Set line thickness for better clarity  
    palette='viridis'  # Use Viridis colour palette for better distinction  
)  

# Set title for the plot  
plt.title('Top 5 High-Energy Nodes: Hourly Patterns', pad=20)  

# Label the x-axis  
plt.xlabel('Hour of the Day', fontsize=12)  

# Label the y-axis  
plt.ylabel('Average Watts', fontsize=12)  

# Customise legend placement and style  
plt.legend(  
    title='Node ID',  # Legend title  
    bbox_to_anchor=(1.05, 1),  # Position legend outside the plot  
    loc='upper left',  # Align legend to the top left  
    frameon=False  # Remove legend box border  
)  

# Add light grid for better readability  
plt.grid(alpha=0.3)  

# Adjust layout to prevent overlapping elements  
plt.tight_layout()  

# Display the plot  
plt.show()  


#%% 5. Modeling Section
# Prepare feature matrix X by dropping target and non-predictive columns
X = df_copy.drop(columns=['watts', 'timestamp', 'day_type'])
# Define target variable y as the energy consumption ('watts')
y = df_copy['watts']

# Initialise StandardScaler to normalise feature values
scaler = StandardScaler()
# Fit the scaler to X and transform it to produce standardised features
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets (80% train, 20% test), with reproducible shuffling.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#%% 5.1 Random Forest
# Initialise Random Forest Regressor model with fixed random seed.
rf = RandomForestRegressor(random_state=42)
# Train the model on the training set.
rf.fit(X_train, y_train)

# Feature Importance
# Plot the importance of each feature in predicting 'watts'
pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot(kind='barh')
plt.title('Random Forest Feature Importance')
plt.show()

# Evaluation
# Use the trained model to predict 'watts' on the test set
y_pred = rf.predict(X_test)
# Print performance metrics: Mean Squared Error and R-squared score
print(f"Random Forest Performance:\nMSE: {mean_squared_error(y_test, y_pred):.2f}\nR2: {r2_score(y_test, y_pred):.2f}")

#%% 5.2 KMeans Clustering 
# Step 1
#The Elbow Method helps visually determine the ideal number of clusters (k)
# by plotting inertia (how tightly the data points are clustered).

# List to store inertia (cluster compactness) for each k
inertia = []
# Try k values from 1 to 10
k_range = range(1, 11)  # Test k=1 to k=10

# For each k, run KMeans and store the inertia
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve to visually identify optimal k
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Auto-detect elbow point using KneeLocator downloaded in the terminal.
# This provides a more objective choice of k based on curvature in the Elbow plot.
try:
    from kneed import KneeLocator
    kneedle = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow
    print(f"Optimal k (auto-detected): {optimal_k}")
except ImportError:
    optimal_k = 4  # Fallback if kneed not installed
    print("Install 'kneed' for auto-detection. Using default k=4.")

#Optimal k (auto-detected): 4

#Step 2: Fit KMeans with the selected optimal number of clusters
# Cluster labels are stored in the dataframe for further analysis/visualisation
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_copy['cluster'] = kmeans.fit_predict(X_scaled)

# Visualise the clusters using the first two scaled features
# Color-coded by cluster label
plt.figure(figsize=(12, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df_copy['cluster'], 
                palette='viridis', alpha=0.7)
plt.title(f'Energy Consumption Clusters (K={optimal_k})')
plt.show()

#%% 5.3 PCA Analysis

# Initialise PCA to reduce features to 2 principal components
pca = PCA(n_components=2)

# Fit and transform the scaled data to obtain principal components
X_pca = pca.fit_transform(X_scaled)

# Print the proportion of variance explained by PC1 and PC2
print(f"PCA Explained Variance: {pca.explained_variance_ratio_}")

# Visualise the data in PCA space, colored by weekday/weekend
plt.figure(figsize=(12, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_copy['day_type'],
                palette={'Weekday': 'blue', 'Weekend': 'red'}, alpha=0.6)
plt.title('PCA: Weekday vs Weekend Patterns')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.show()

#%% 6. Advanced Post-PCA Analysis

# Initialise Isolation Forest for anomaly detection (5% contamination)
iso = IsolationForest(contamination=0.05, random_state=42)

# Fit the model and label anomalies in the dataset
df_copy['anomaly'] = iso.fit_predict(X_scaled)

# Plot PCA points with anomalies highlighted
plt.figure(figsize=(12, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], 
                hue=df_copy['anomaly'].map({1: 'Normal', -1: 'Anomaly'}), 
                style=df_copy['day_type'],
                palette={'Normal': 'green', 'Anomaly': 'red'})
plt.title('PCA with Anomaly Detection')
plt.show()

#%% PCA Loadings

# Extract feature contributions to each principal component
loadings = pd.DataFrame(
    pca.components_.T,  # Transpose to align features with PCs
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=X.columns  # Use original feature names as index
)
print("PCA Loadings:\n", loadings)

#%% Add PCA scores to the main DataFrame for further analysis

# Add PC1 values to the DataFrame
df_copy['PC1'] = X_pca[:, 0]
# Compare average PC1 values across weekend vs weekday
print(df_copy.groupby('is_weekend')['PC1'].mean())

# Add PC2 values to the DataFrame
df_copy['PC2'] = X_pca[:, 1]
# Compare average PC2 values across hours of the day
print(df_copy.groupby('hour')['PC2'].mean().sort_values())

#%% PCA Driver Features

# List features with the strongest influence on PC1
print("PC1 Drivers (Weekend/Node):\n", loadings['PC1'].abs().sort_values(ascending=False))

# List features with the strongest influence on PC2
print("\nPC2 Drivers (Hourly/Weekly):\n", loadings['PC2'].abs().sort_values(ascending=False))

#%% 7. Final Insights

print("\n=== Key Insights ===")

# Compare average watts between weekend and weekday
print(f"- Weekend consumption is {df_copy.groupby('is_weekend')['watts'].mean().diff().iloc[-1]:.1f} watts higher than weekdays")

# Show dates of identified anomaly days
print(f"- Top anomaly days: {df_copy[df_copy['anomaly'] == -1]['timestamp'].dt.date.unique()}")

# Display the most important feature from the Random Forest
print(f"- Most important feature: {X.columns[np.argmax(rf.feature_importances_)]}")

# Recap PCA insights and variance explained
print(f"- PCA separates weekdays (PC1<0) from weekends (PC1>0):\n  - PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance\n  - PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% variance")