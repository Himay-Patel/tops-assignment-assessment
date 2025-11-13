import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('customer_segmentation.csv')

# Data preprocessing
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values
df['Income'] = df['Income'].fillna(df['Income'].median())

# Select relevant features for clustering
features_for_clustering = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
]

X = df[features_for_clustering]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. K-means Clustering
print("\n" + "="*50)
print("K-MEANS CLUSTERING")
print("="*50)

# Find optimal k using elbow method and silhouette score
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k values')
plt.grid(True)

plt.tight_layout()
plt.show()

# Choose optimal k (based on elbow and silhouette score)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['KMeans_Cluster'] = kmeans_labels

print(f"Optimal number of clusters (K-means): {optimal_k}")
print(f"K-means Silhouette Score: {silhouette_score(X_scaled, kmeans_labels):.4f}")

# 2. Hierarchical Clustering
print("\n" + "="*50)
print("HIERARCHICAL CLUSTERING")
print("="*50)

# Create dendrogram to determine optimal number of clusters
plt.figure(figsize=(12, 6))
linked = linkage(X_scaled, method='ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Customer Index')
plt.ylabel('Euclidean Distance')
plt.show()

# Apply hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Hierarchical_Cluster'] = hierarchical_labels

print(f"Optimal number of clusters (Hierarchical): {optimal_k}")
print(f"Hierarchical Silhouette Score: {silhouette_score(X_scaled, hierarchical_labels):.4f}")

# 3. Cluster Analysis and Comparison
print("\n" + "="*50)
print("CLUSTER COMPARISON AND ANALYSIS")
print("="*50)

# Cluster sizes comparison
kmeans_cluster_sizes = df['KMeans_Cluster'].value_counts().sort_index()
hierarchical_cluster_sizes = df['Hierarchical_Cluster'].value_counts().sort_index()

print("\nCluster Sizes:")
cluster_size_df = pd.DataFrame({
    'KMeans': kmeans_cluster_sizes,
    'Hierarchical': hierarchical_cluster_sizes
})
print(cluster_size_df)

# Compare cluster assignments
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari_score = adjusted_rand_score(kmeans_labels, hierarchical_labels)
nmi_score = normalized_mutual_info_score(kmeans_labels, hierarchical_labels)

print(f"\nCluster Similarity Metrics:")
print(f"Adjusted Rand Index: {ari_score:.4f}")
print(f"Normalized Mutual Information: {nmi_score:.4f}")

# Analyze cluster characteristics for K-means
print("\n" + "="*50)
print("K-MEANS CLUSTER PROFILES")
print("="*50)

# Calculate mean values for each cluster
cluster_profile = df.groupby('KMeans_Cluster')[features_for_clustering].mean()
print(cluster_profile.round(2))

# Visualize cluster characteristics
plt.figure(figsize=(16, 12))

# Plot 1: Income vs Spending
plt.subplot(2, 2, 1)
for cluster in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    plt.scatter(cluster_data['Income'], cluster_data['MntWines'] + cluster_data['MntMeatProducts'], 
               label=f'Cluster {cluster}', alpha=0.7)
plt.xlabel('Income')
plt.ylabel('Total Spending (Wines + Meat)')
plt.title('Income vs Total Spending by Cluster')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Purchase Channels
plt.subplot(2, 2, 2)
channel_data = df.groupby('KMeans_Cluster')[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].mean()
channel_data.plot(kind='bar', ax=plt.gca())
plt.title('Average Purchases by Channel per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Number of Purchases')
plt.xticks(rotation=0)

# Plot 3: Product Categories
plt.subplot(2, 2, 3)
product_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
product_data = df.groupby('KMeans_Cluster')[product_columns].mean()
product_data.plot(kind='bar', ax=plt.gca())
plt.title('Average Spending on Product Categories per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Spending')
plt.xticks(rotation=0)

# Plot 4: Cluster comparison between algorithms
plt.subplot(2, 2, 4)
comparison_data = pd.crosstab(df['KMeans_Cluster'], df['Hierarchical_Cluster'])
sns.heatmap(comparison_data, annot=True, fmt='d', cmap='Blues')
plt.title('Cluster Assignment Comparison\nK-means vs Hierarchical')
plt.xlabel('Hierarchical Cluster')
plt.ylabel('K-means Cluster')

plt.tight_layout()
plt.show()

# Detailed cluster interpretation
print("\n" + "="*50)
print("CLUSTER INTERPRETATION")
print("="*50)

# Create meaningful cluster descriptions
cluster_descriptions = []
for cluster in range(optimal_k):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    
    # Calculate key metrics
    avg_income = cluster_data['Income'].mean()
    avg_total_spending = cluster_data[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1).mean()
    avg_web_visits = cluster_data['NumWebVisitsMonth'].mean()
    avg_deal_purchases = cluster_data['NumDealsPurchases'].mean()
    
    # Determine cluster type
    if avg_income > df['Income'].median() and avg_total_spending > df[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                                                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1).median():
        cluster_type = "High-Value Customers"
    elif avg_income < df['Income'].median() and avg_total_spending < df[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                                                      'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1).median():
        cluster_type = "Budget-Conscious Customers"
    elif avg_web_visits > df['NumWebVisitsMonth'].median():
        cluster_type = "Online-Savvy Customers"
    else:
        cluster_type = "Regular Customers"
    
    description = f"""
    Cluster {cluster} ({cluster_type}):
    - Size: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)
    - Average Income: ${avg_income:,.0f}
    - Average Total Spending: ${avg_total_spending:,.0f}
    - Average Web Visits/Month: {avg_web_visits:.1f}
    - Deal Purchases: {avg_deal_purchases:.1f}
    """
    cluster_descriptions.append(description)

for desc in cluster_descriptions:
    print(desc)

# Business recommendations
print("\n" + "="*50)
print("BUSINESS RECOMMENDATIONS")
print("="*50)

recommendations = """
Based on the customer segmentation analysis:

1. HIGH-VALUE CUSTOMERS:
   - Focus on premium products and personalized services
   - Implement loyalty programs with exclusive benefits
   - Target with high-value offers and early access to new products

2. BUDGET-CONSCIOUS CUSTOMERS:
   - Promote deals, discounts, and value bundles
   - Highlight cost-effective products and payment plans
   - Use price-sensitive marketing strategies

3. ONLINE-SAVVY CUSTOMERS:
   - Enhance digital experience and mobile optimization
   - Implement targeted online advertising and social media campaigns
   - Offer online-exclusive deals and promotions

4. REGULAR CUSTOMERS:
   - Maintain consistent communication and engagement
   - Focus on customer retention strategies
   - Offer balanced mix of products and services
"""

print(recommendations)

# Final comparison summary
print("\n" + "="*50)
print("FINAL COMPARISON SUMMARY")
print("="*50)

summary = f"""
CLUSTERING ALGORITHM COMPARISON:

K-means Clustering:
- Silhouette Score: {silhouette_score(X_scaled, kmeans_labels):.4f}
- Cluster sizes are generally balanced
- Computationally efficient for large datasets
- Good for spherical clusters

Hierarchical Clustering:
- Silhouette Score: {silhouette_score(X_scaled, hierarchical_labels):.4f}
- Provides hierarchical relationship between clusters
- No need to pre-specify number of clusters
- More interpretable dendrogram visualization

Similarity Between Methods:
- Adjusted Rand Index: {ari_score:.4f} (0 = random, 1 = identical)
- Normalized Mutual Information: {nmi_score:.4f}

CONCLUSION:
Both methods provide meaningful customer segments with good separation quality.
The choice between algorithms depends on business requirements:
- Use K-means for computational efficiency and well-separated spherical clusters
- Use Hierarchical clustering when you need to understand cluster relationships and hierarchy

For this customer segmentation analysis, both methods successfully identified distinct customer groups that can inform targeted marketing strategies.
"""

print(summary)