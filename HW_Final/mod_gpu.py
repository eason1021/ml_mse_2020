import cupy as cp
import cudf as cd
import pandas as pd
import cugraph as cg
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

read_file_starttime = datetime.datetime.now()

XYZ_C = cd.read_csv('HW_AI/HW_Final/dataset/pos_50.csv', index_col = 0).to_numpy()

XYZ_Edges = cd.read_csv('HW_AI/HW_Final/dataset/edges_50.csv', index_col = 0,dtype=['int32', 'int32', 'int32','float32','str'])

read_file_endtime = datetime.datetime.now()
print (read_file_endtime - read_file_starttime)

graph_build_starttime = datetime.datetime.now()

G = cg.Graph()
G = cg.from_cudf_edgelist(XYZ_Edges, source = 'Source', destination = 'Target', edge_attr = 'Weight')
graph_build_endtime = datetime.datetime.now()
print (graph_build_endtime - graph_build_starttime)

louvain_starttime = datetime.datetime.now()
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
louvain_endtime = datetime.datetime.now()
print(louvain_endtime - louvain_starttime)
print(labelRE)
labelRE = cp.asfortranarray(labelRE)
labelRE = cd.from_dlpack(labelRE.toDlpack())
#labelRE = cd.DataFrame(labelRE)
print(labelRE)
df = cd.DataFrame({'label': labelRE})
df.to_csv('HW_AI/HW_Final/gpu.csv')

'''
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
plt.show()'''