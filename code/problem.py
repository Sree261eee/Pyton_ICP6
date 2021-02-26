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

#read csv file
dataset = pd.read_csv("CC.csv")
# drop cust_id because it doesot has any relationship with features
dataset = dataset.drop(['CUST_ID'], axis=1)
#replacing null values with mean
print(dataset.isnull().sum()) # checking null values in each column
dataset = dataset.fillna(dataset.mean())


#elbow method to k value
x = dataset.iloc[:,[i for i in range(16)]] # selecting features from dataset
y = dataset.iloc[:,-1] # cluster column
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
#from the graph plotted we can take k=3

#silhouette score calculation
nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x) #training the data
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score) # accuracy

