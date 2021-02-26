import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

# Reading csv file
dataset = pd.read_csv("CC.csv")
# drop cust_id because it doesot has any relationship with features
dataset = dataset.drop(['CUST_ID'], axis=1)
#replacing null values with mean
dataset = dataset.fillna(dataset.mean())


x = dataset.iloc[:,[i for i in range(16)]] #splitting features
y = dataset.iloc[:,-1]

#prepocessing of data
scaler = preprocessing.StandardScaler() #scaling is done to handle highly varying magnitudes or values
scaler.fit(x)
X_scaled_array = scaler.transform(x) #transforming the values
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns) #dataframe of transformed values

#finding k-value using elbow method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

#plotting elbow graph
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

#from the graph we cam assume k=8
#I have tested for different k-values from 5 to 9 at k=8 I got good accuracy
nclusters = 8 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(X_scaled)
yscaled_cluster_kmeans = km.predict(X_scaled)
scaled_score = metrics.silhouette_score(X_scaled, yscaled_cluster_kmeans)
print(scaled_score)