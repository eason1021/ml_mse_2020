import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


label_result = pd.read_csv('HW_AI/HW_Final/label.csv')
label_result = label_result.iloc[:,1]
label_result = label_result.to_numpy()
cpu_result = pd.read_csv('HW_AI/HW_Final/cpu.csv')
cpu_result = cpu_result.iloc[:,1]
cpu_result = cpu_result.to_numpy()
cpu_result = cpu_result.astype(np.int32)
gpu_result = pd.read_csv('HW_AI/HW_Final/gpu.csv')
gpu_result = gpu_result['label'].to_numpy()

label_cpu_mutual = metrics.adjusted_mutual_info_score(label_result, cpu_result)
label_gpu_mutual = metrics.adjusted_mutual_info_score(label_result, gpu_result)
cpu_gpu_mutual = metrics.adjusted_mutual_info_score(cpu_result, gpu_result)
print(cpu_result)
print(gpu_result)
print(label_cpu_mutual)
print(label_gpu_mutual)
print(cpu_gpu_mutual)

cpu_time = [1.4,14.5,1170.2]
gpu_time = [0.97, 0.16, 0.32]

x = [1,2]
read = [1.4, 0.97]
build_graph = [15.9, 1.13]
louvain = [1186.1, 1.45]

plt.barh(x, read, label = 'Read_Files')
plt.barh(x, build_graph, label = 'Build Graph')
plt.barh(x, louvain, label = 'Louvain_Modularity')
plt.yticks([1,2],['Cpu', 'Gpu'])
plt.title('Time compare')
plt.xlabel('Time')
plt.ylabel('Device')
plt.legend(loc = 'best')
plt.show()