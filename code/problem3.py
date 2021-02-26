import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv("CC.csv")
# drop cust_id because it doesot has any relationship with features
dataset = dataset.drop(['CUST_ID'], axis=1)
#replacing null values with mean
dataset = dataset.fillna(dataset.mean())
# print(dataset.isnull().sum())


x = dataset.iloc[:,[i for i in range(16)]] #splitting features
y = dataset.iloc[:,-1]

#scaling data
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x)

# Apply transformation
x_scaler = scaler.transform(x)
pca = PCA(2) #decreasing columns for better accuracy
x_pca = pca.fit_transform(x_scaler) #transformation after PCA

#finding k-means using elbow method for PCA values
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

#Dataframe for PCA data
df = pd.DataFrame(data=x_pca, columns=["x", "y"])
finaldf = pd.concat([df,dataset[['TENURE']]],axis=1)
#for k=5 we got the better accuracy
nclusters = 5 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(df)
plt.scatter(df["x"], df["y"])
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=300, c='red')
plt.show()
yscaled_cluster_kmeans = km.predict(df)
scaled_score = metrics.silhouette_score(df, yscaled_cluster_kmeans)
print(scaled_score)