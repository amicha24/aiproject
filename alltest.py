from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score,silhouette_samples
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Wine data set
df = pd.read_csv('recipeData.csv', encoding='latin1')
df=df[['ABV','IBU', 'Color']]

y = []
matrix = df.to_numpy()
for n_clusters in range(2,10):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=10)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    y.append(silhouette_avg)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

plt.figure(figsize=(12,8))
plt.plot(range(2,10),y)
plt.xlabel('No of Clusters')
plt.ylabel('Silhouette_avg')
plt.title('Silhoutte Score for different clusters')




df_missing = df.copy()
df_missing = df_missing.T
true = df_missing.isnull().sum(axis=1)
false = (len(df_missing.columns) - true)
df_missing['Valid Count'] = false / len(df_missing.columns)
df_missing['NA Count'] = true / len(df_missing.columns)

df_missing[['NA Count','Valid Count']].sort_values(
    'NA Count', ascending=False).plot.bar(
    stacked=True,figsize=(12,6))
plt.legend(loc=9)
plt.ylim(0,1.15)
plt.title('Normed Missing Values Count', fontsize=20)
plt.xlabel('Normed (%) count', fontsize=20)
plt.ylabel('Column name', fontsize=20)
plt.xticks(rotation=60)
plt.show()

#df = df[pd.notnull(df['Style'])] #use only samples with valid Style col


#df = df.dropna(subset=['ABV', 'IBU', 'Color'])

df_abv_color = df[(df['ABV']<=20) & (df['Color']<=50)]
df_abv_color = df_abv_color.sample(int(len(df_abv_color)/10), random_state=42)

plt.figure(figsize=(12,6))
sns.regplot(df_abv_color['ABV'],df_abv_color['Color'])
plt.title('ABV and Color relation', fontsize=22)
plt.xlabel('ABV', fontsize=20)
plt.ylabel('Color', fontsize=20)
plt.show()

# Generate clusters from K-Means
km = KMeans(3)
km_clusters = km.fit_predict(df)


db_param_options = [[20,5],[25,5],[30,5],[25,7],[35,7],[35,3]]

for ep,min_sample in db_param_options:
    # Generate clusters using DBSCAN
    db = DBSCAN(eps=ep, min_samples = min_sample)
    db_clusters = db.fit_predict(df)
    print("Eps: ", ep, "Min Samples: ", min_sample)
    print("DBSCAN Clustering: ", silhouette_score(df, db_clusters))


# Generate clusters using DBSCAN
db = DBSCAN(eps=35, min_samples = 3)
db_clusters = db.fit_predict(df)

plt.title("Beer Clusters from K-Means")
plt.scatter(df['ABV'], df['Color'], c=km_clusters,s=50, cmap='tab20b')
plt.show()

plt.title("Beer Clusters from DBSCAN")
plt.scatter(df['ABV'], df['Color'], c=db_clusters,s=50, cmap='tab20b')
plt.show()

# Calculate Silhouette Scores
print("Silhouette Scores for Beer Dataset:\n")
print("K-Means Clustering: ", silhouette_score(df, km_clusters))
print("DBSCAN Clustering: ", silhouette_score(df, db_clusters))
