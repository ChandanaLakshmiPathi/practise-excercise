#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Load the Dataset and Check for Missing Values:
import pandas as pd

# Load the dataset
data = pd.read_excel('Online Retail.xlsx', engine='openpyxl')

# Clean the data by removing rows with missing CustomerID and Description
cleaned_data = data.dropna(subset=['CustomerID', 'Description'])

# Check for missing values
missing_values = cleaned_data.isnull().sum()
print("Missing Values:\n", missing_values)

# Display the shape of the dataset
print("Shape of the dataset:", cleaned_data.shape)


# In[14]:


pip install pandas openpyxl


# In[4]:


import pandas as pd

# Load the dataset
data = pd.read_excel('Online Retail.xlsx', engine='openpyxl')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Check the shape of the dataset
print("Shape of the dataset:", data.shape)

# Display basic information about the dataset
print(data.info())


# In[9]:


#Visualize Feature Distributions
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate total spending per customer
cleaned_data['TotalSpending'] = cleaned_data['Quantity'] * cleaned_data['UnitPrice']

# Group by CustomerID to get total spending and number of purchases
customer_data = cleaned_data.groupby('CustomerID').agg(
    TotalSpending=('TotalSpending', 'sum'),
    NumberOfPurchases=('Quantity', 'sum')
).reset_index()

# Visualize Total Spending vs. Number of Purchases
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_data, x='TotalSpending', y='NumberOfPurchases')
plt.title('Total Spending vs. Number of Purchases')
plt.xlabel('Total Spending ($)')
plt.ylabel('Number of Purchases')
plt.grid()
plt.show()


# In[10]:


# Correlation analysis
correlation_matrix = customer_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Implement K-Means Clustering

# In[11]:


#Elbow Method to Determine the Optimal Number of Clusters:


from sklearn.cluster import KMeans

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customer_data[['TotalSpending', 'NumberOfPurchases']])
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid()
plt.show()


# In[12]:


#Apply Silhouette Score to Validate Cluster Quality:
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in k_values[1:]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customer_data[['TotalSpending', 'NumberOfPurchases']])
    score = silhouette_score(customer_data[['TotalSpending', 'NumberOfPurchases']], kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(k_values[1:], silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()


# In[13]:


#Visualize Customer Segments:
# Assuming k=3 is chosen based on the elbow method
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data[['TotalSpending', 'NumberOfPurchases']])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_data, x='TotalSpending', y='NumberOfPurchases', hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Total Spending ($)')
plt.ylabel('Number of Purchases')
plt.grid()
plt.show()


# In[20]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load and clean the dataset
data = pd.read_excel('Online Retail.xlsx', engine='openpyxl')
cleaned_data = data.dropna(subset=['CustomerID', 'Description'])
cleaned_data['TotalSpending'] = cleaned_data['Quantity'] * cleaned_data['UnitPrice']
customer_data = cleaned_data.groupby('CustomerID').agg(TotalSpending=('TotalSpending', 'sum'),
                                                        NumberOfPurchases=('Quantity', 'sum')).reset_index()

# Feature Scaling
scaled_data = StandardScaler().fit_transform(customer_data[['TotalSpending', 'NumberOfPurchases']])

# PCA
pca_data = PCA(n_components=2).fit_transform(scaled_data)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_data)

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='viridis')
plt.title('Customer Segments after PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid()
plt.show()


# In[ ]:




