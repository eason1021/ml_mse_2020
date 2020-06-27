from ovito.io import import_file
from ovito.io import export_file
from ovito.modifiers import *
from ovito.data import *
from ovito.data import NearestNeighborFinder
import numpy as np
import pandas as pd
import sys
import math
import os
import random
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset

my_var = {}
file_path = '/data2/mphase/set/average_50_33.dump'
node = import_file(file_path)
node_1 = import_file(file_path)
node_1.modifiers.append(SelectParticleTypeModifier(property='Particle Type', types={2}))
node_1.modifiers.append(DeleteSelectedParticlesModifier())
node_2 = import_file(file_path)
node_2.modifiers.append(SelectParticleTypeModifier(property='Particle Type', types={1}))
node_2.modifiers.append(DeleteSelectedParticlesModifier())

ref_path = '/data2/mphase/ref.dump'
node_ref = import_file(ref_path)
node_ref_1 = import_file(ref_path)
node_ref_1.modifiers.append(SelectParticleTypeModifier(property='Particle Type', types={2}))
node_ref_1.modifiers.append(DeleteSelectedParticlesModifier())
node_ref_2 = import_file(ref_path)
node_ref_2.modifiers.append(SelectParticleTypeModifier(property='Particle Type', types={1}))
node_ref_2.modifiers.append(DeleteSelectedParticlesModifier())

def ref_neighbor(data):
	ref_list = []
	N = 6
	finder = NearestNeighborFinder(N, data)
	ptype = data.particles['Particle Type']
	
	#Loop over all input particles:
	for index in range(data.particles.count):
		neighbors = [ (neigh.index, neigh.delta) for neigh in finder.find(index) ]
		neigh_list = [0]*6
		resorted_neighbors_x_ref = sorted( neighbors , key=lambda k: [k[1][0], k[1][1], k[1][2]], reverse=True )
		resorted_neighbors_y_ref = sorted( neighbors , key=lambda k: [k[1][1], k[1][0], k[1][2]], reverse=True )
		resorted_neighbors_z_ref = sorted( neighbors , key=lambda k: [k[1][2], k[1][0], k[1][1]], reverse=True )
		resorted_neighbors_nx_ref = sorted( neighbors , key=lambda k: [k[1][0], k[1][1], k[1][2]], reverse=False )
		resorted_neighbors_ny_ref = sorted( neighbors , key=lambda k: [k[1][1], k[1][0], k[1][2]], reverse=False )
		resorted_neighbors_nz_ref = sorted( neighbors , key=lambda k: [k[1][2], k[1][0], k[1][1]], reverse=False )
		neigh_list[0] = resorted_neighbors_x_ref[0]
		neigh_list[1] = resorted_neighbors_y_ref[0]
		neigh_list[2] = resorted_neighbors_z_ref[0]
		neigh_list[3] = resorted_neighbors_nx_ref[0]
		neigh_list[4] = resorted_neighbors_ny_ref[0]
		neigh_list[5] = resorted_neighbors_nz_ref[0]
		my_var["neigh_list%s"%index] = neigh_list

	for neigh_sort in range(data.particles.count):
		phase_index_list = [0]*8
		#+X
		phase_index_list[0] = my_var["neigh_list%s"%neigh_sort][0]
		#+Y
		phase_index_list[1] = my_var["neigh_list%s"%neigh_sort][1]
		#+Z
		phase_index_list[2] = my_var["neigh_list%s"%neigh_sort][2]
		#-X
		phase_index_list[3] = my_var["neigh_list%s"%neigh_sort][3]
		#-Y
		phase_index_list[4] = my_var["neigh_list%s"%neigh_sort][4]
		#-Z
		phase_index_list[5] = my_var["neigh_list%s"%neigh_sort][5]
		#+X+Z
		phase_index_list[6] = (my_var["neigh_list%s"%my_var['neigh_list%s'%neigh_sort][0][0]][2][0],tuple(((np.array(my_var['neigh_list%s'%my_var['neigh_list%s'%neigh_sort][0][0]][2][1]) + np.array((my_var['neigh_list%s'%neigh_sort][0][1]))).tolist())))
		#+Y+Z
		phase_index_list[7] = (my_var["neigh_list%s"%my_var['neigh_list%s'%neigh_sort][1][0]][2][0],tuple(((np.array(my_var['neigh_list%s'%my_var['neigh_list%s'%neigh_sort][1][0]][2][1]) + np.array((my_var['neigh_list%s'%neigh_sort][1][1]))).tolist())))

		ref_list.append(phase_index_list)
	return ref_list

def edges(frame, data, ref):
	property_list = []
	graph_list = []
	content_list = []
	N = 6
	finder = NearestNeighborFinder(N, data)
	pid = data.particles['Particle Identifier']
	pos = data.particles['Position']
	#Loop over all input particles:
	for index in range(data.particles.count):
		neighbors = [ (neigh.index, neigh.delta) for neigh in finder.find(index) ]
		neigh_list = [0]*6
		resorted_neighbors_x_ref = sorted( neighbors , key=lambda k: [k[1][0], k[1][1], k[1][2]], reverse=True )
		resorted_neighbors_y_ref = sorted( neighbors , key=lambda k: [k[1][1], k[1][0], k[1][2]], reverse=True )
		resorted_neighbors_z_ref = sorted( neighbors , key=lambda k: [k[1][2], k[1][0], k[1][1]], reverse=True )
		resorted_neighbors_nx_ref = sorted( neighbors , key=lambda k: [k[1][0], k[1][1], k[1][2]], reverse=False )
		resorted_neighbors_ny_ref = sorted( neighbors , key=lambda k: [k[1][1], k[1][0], k[1][2]], reverse=False )
		resorted_neighbors_nz_ref = sorted( neighbors , key=lambda k: [k[1][2], k[1][0], k[1][1]], reverse=False )
		neigh_list[0] = resorted_neighbors_x_ref[0]
		neigh_list[1] = resorted_neighbors_y_ref[0]
		neigh_list[2] = resorted_neighbors_z_ref[0]
		neigh_list[3] = resorted_neighbors_nx_ref[0]
		neigh_list[4] = resorted_neighbors_ny_ref[0]
		neigh_list[5] = resorted_neighbors_nz_ref[0]
		my_var["neigh_list%s"%index] = neigh_list

	for neigh_sort in range(data.particles.count):
		phase_index_list = [0]*8
		#+X
		phase_index_list[0] = my_var["neigh_list%s"%neigh_sort][0]
		#+Y
		phase_index_list[1] = my_var["neigh_list%s"%neigh_sort][1]
		#+Z
		phase_index_list[2] = my_var["neigh_list%s"%neigh_sort][2]
		#-X
		phase_index_list[3] = my_var["neigh_list%s"%neigh_sort][3]
		#-Y
		phase_index_list[4] = my_var["neigh_list%s"%neigh_sort][4]
		#-Z
		phase_index_list[5] = my_var["neigh_list%s"%neigh_sort][5]
		#+X+Z
		phase_index_list[6] = (my_var["neigh_list%s"%my_var['neigh_list%s'%neigh_sort][0][0]][2][0],tuple(((np.array(my_var['neigh_list%s'%my_var['neigh_list%s'%neigh_sort][0][0]][2][1]) + np.array((my_var['neigh_list%s'%neigh_sort][0][1]))).tolist())))
		#+Y+Z
		phase_index_list[7] = (my_var["neigh_list%s"%my_var['neigh_list%s'%neigh_sort][1][0]][2][0],tuple(((np.array(my_var['neigh_list%s'%my_var['neigh_list%s'%neigh_sort][1][0]][2][1]) + np.array((my_var['neigh_list%s'%neigh_sort][1][1]))).tolist())))

		reference = ref[neigh_sort]
		for i in range(len(phase_index_list)):
			graph_list.append(pid[neigh_sort])
			graph_list.append(pid[phase_index_list[i][0]])

			delta_x = np.array(phase_index_list[i][1][0])-np.array(reference[i][1][0])
			delta_y = np.array(phase_index_list[i][1][1])-np.array(reference[i][1][1])
			delta_z = np.array(phase_index_list[i][1][2])-np.array(reference[i][1][2])
			distance = np.sqrt((delta_x**2+delta_y**2+delta_z**2))
			content_list.append(1 / distance)
	result = (graph_list, content_list)
	return result

ref_list_type1 = ref_neighbor(node_ref_1.compute())
result_1 = edges(0, node_1.compute(0), ref_list_type1)

ref_list_type2 = ref_neighbor(node_ref_2.compute())
result_2 = edges(0, node_2.compute(0), ref_list_type2)

graph_list_1 = np.array(result_1[0])
content_list_1 = np.array(result_1[1])

graph_list_2 = np.array(result_2[0])
content_list_2 = np.array(result_2[1])

graph_list_1 = np.reshape(graph_list_1, (-1, 2))
graph_list_2 = np.reshape(graph_list_2, (-1, 2))
graph = np.row_stack((graph_list_1,graph_list_2))

content_list_1 = np.reshape(content_list_1, (-1, 1))
content_list_2 = np.reshape(content_list_2, (-1, 1))
content = np.row_stack((content_list_1,content_list_2))

data = np.column_stack((graph, content))
data = np.unique(data, axis=0)
source = data[:, 0]
target = data[:, 1]
weight = data[:, 2]
type_array = ['Undirected'] * data.shape[0]

df = pd.DataFrame({"Source" : source,
					"Target" : target,
					"Weight" : weight,
					"Type" : type_array})
df.to_csv('HW_AI/HW_Final/dataset/edges_50.csv')

posi = node.compute(0).particles['Position']
df_2 = pd.DataFrame({"X" : posi[:,0],
					"Y" : posi[:,1], 
					"Z" : posi[:,2]})
df_2.to_csv('HW_AI/HW_Final/dataset/pos_50.csv')