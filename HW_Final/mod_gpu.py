import cupy as cp
import cudf as cd
import pandas as pd
import cugraph as cg
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

starttime = datetime.datetime.now()

XYZ_C = pd.read_csv('HW_AI/HW_Final/dataset/pos_50.csv', index_col = 0).to_numpy()

XYZ_Edges = cd.read_csv('HW_AI/HW_Final/dataset/edges_50.csv', index_col = 0,dtype=['int32', 'int32', 'int32','float32','str'])

G = cg.Graph()
G = cg.from_cudf_edgelist(XYZ_Edges, source = 'Source', destination = 'Target', edge_attr = 'Weight')
result, mod = cg.louvain(G)
vertex = result['vertex']
partition = result['partition']
size = result['partition'].max() + 1
print('community', size)
print('modularity', mod)
vertex = cp.fromDlpack(vertex.to_dlpack())
partition = cp.fromDlpack(partition.to_dlpack())
vertex = cp.reshape(vertex, XYZ_C.shape[0])
labelRE = cp.reshape(partition, XYZ_C.shape[0])
index = cp.argsort(vertex)
vertex = cp.take_along_axis(vertex, index, axis=0)
labelRE = cp.take_along_axis(labelRE, index, axis=0)
print(result)
print(vertex)
print(labelRE)
print(index)
labelRE = cp.asnumpy(labelRE)
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