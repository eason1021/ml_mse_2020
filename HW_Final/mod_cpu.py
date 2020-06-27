import numpy as np
import pandas as pd
import scipy.io
import community
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import metrics
import datetime

starttime = datetime.datetime.now()

XYZ_C = pd.read_csv('HW_AI/HW_Final/dataset/pos_50.csv', index_col = 0).to_numpy()
print(XYZ_C)

XYZ_Edges = pd.read_csv('HW_AI/HW_Final/dataset/edges_50.csv', index_col = 0)
print(XYZ_Edges)

XYZ_E = np.array(XYZ_Edges.iloc[:, 0:3].values)
print(XYZ_E)

#Use pre-defined linkage (Edges.csv) to constructure whole network
G = nx.Graph()
for i in range(0, len(XYZ_E)):    
    e = ( str(int(XYZ_E[i,0])), str(int(XYZ_E[i,1])), XYZ_E[i,2] )
    G.add_weighted_edges_from([(e)])    

partition = community.best_partition(G)
size = float(len(set(partition.values())))
print("community:", size)
mod = community.modularity(partition,G)
print("modularity:", mod)

#assign data point color based on community in realspace
label = np.zeros((len(XYZ_C),1))
for j in set(partition.values()) :
    for i in range(len(XYZ_C)) :
        if partition[str(i+1)] == j :            
            label[i] =  j
labelRE = np.reshape(label, len(XYZ_C))            

endtime = datetime.datetime.now()
print (endtime - starttime)
plt.figure(1)
ax = plt.axes(projection='3d')
z = XYZ_C[:,2]
x = XYZ_C[:,0]
y = XYZ_C[:,1]
c = labelRE
ax.scatter(x, y, z, c = c, cmap = plt.get_cmap('jet'))
plt.title('Cluster result by modularity')
ax.view_init(80, 0)

plt.figure(2)
ax = plt.axes(projection='3d')
z = XYZ_C[:,2]
x = XYZ_C[:,0]
y = XYZ_C[:,1]
c = labelRE
ax.scatter(x, y, z, c = c, cmap = plt.get_cmap('jet'))
ax.view_init(45, 0)
plt.title('Cluster result by modularity')
plt.show()