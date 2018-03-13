
# coding: utf-8


import pandas as pd

df = pd.read_csv('C:/Users/dycheng/Downloads/data_1024.csv',sep='\t')
##df = pd.read_csv('C:/Users/dycheng/OneDrive-Qualcomm/AI/Jupyter Data/k-mean/data_1024.csv')
#print(df)
df.head()
#print(df['Distance_Feature'])

import numpy as np
import matplotlib 

#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

#plt.ioff()
#plt.plot([1.6, 2.7])
#plt.title("interactive test")
#plt.xlabel("index")
#plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df['Distance_Feature'].values,df['Speeding_Feature'].values,df['Driver_ID'].values,c='b')
ax.set_xlabel('Distance')
ax.set_ylabel('Speed')
ax.set_zlabel('ID')

fig.show()


# In[42]:

import numpy as np
from sklearn.cluster import KMeans

### For the purposes of this example, we store feature data from our
### dataframe `df`, in the `f1` and `f2` arrays. We combine this into
### a feature matrix `X` before entering it into the algorithm.
#print(df['Distance_Feature'].values)
f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values

#print(f1,'\n',f2)
zipped = zip(f1,f2)

#print(list(zipped))
X=np.matrix(list(zipped))
#print (X)

kmeans = KMeans(n_clusters=2).fit(X)
#print(kmeans.labels_)  # result of KMeans is each data point marked with a label
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df['Distance_Feature'].values,df['Speeding_Feature'].values,df['Driver_ID'].values,c=kmeans.labels_.astype(np.float))
ax.set_xlabel('Distance')
ax.set_ylabel('Speed')
ax.set_zlabel('ID')
ax.set_title('n=2')
fig.show()

kmeans = KMeans(n_clusters=4).fit(X)
#print(kmeans.labels_)  # result of KMeans is each data point marked with a label
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df['Distance_Feature'].values,df['Speeding_Feature'].values,df['Driver_ID'].values,c=kmeans.labels_.astype(np.float))
ax.set_xlabel('Distance')
ax.set_ylabel('Speed')
ax.set_zlabel('ID')
ax.set_title('n=4')
fig.show()
