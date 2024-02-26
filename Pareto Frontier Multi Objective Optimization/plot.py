import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import os
import glob


file_names = glob.glob("results/movielens1m/*.tsv")

fig = plt.figure(figsize=(14, 6))
ax2d = fig.add_subplot(121)
model_names_2d = []  
for file_name in file_names:
    if 'nDCG_APLT' in file_name:
        df = pd.read_csv(file_name, sep='\t')
        model_name = os.path.basename(file_name).split('_')[1]  
        ax2d.scatter(df['nDCG'], df['APLT'], label=model_name)
        model_names_2d.append(model_name)

ax2d.set_xlabel('nDCG')
ax2d.set_ylabel('APLT')
ax2d.set_title('Movielens1M, nDCG/APLT')
if model_names_2d:  
    ax2d.legend()


ax3d = fig.add_subplot(122, projection='3d')
model_names_3d = []  
for file_name in file_names:
    if 'nDCG_Gini_EPC' in file_name:
        df = pd.read_csv(file_name, sep='\t')
        model_name = os.path.basename(file_name).split('_')[1]  
        ax3d.scatter(df['nDCG'], df['Gini'], df['EPC'], label=model_name)
        model_names_3d.append(model_name)

ax3d.set_xlabel('nDCG')
ax3d.set_ylabel('Gini')
ax3d.set_zlabel('EPC')
ax3d.set_title('Movielens1M, nDCG/Gini/EPC')
if model_names_3d:  
    ax3d.legend()

plt.tight_layout()
plt.show()
