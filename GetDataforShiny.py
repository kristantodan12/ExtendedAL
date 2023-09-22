"""
This code aims to generate data for shiny app visualization. The outputs will be dataframes of nodes and links
that need to be copied to the R directory to generate the app
"""

# %% Set environment
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import os
import networkx as nx
# %%Reading data
PROJECT_ROOT = Path.cwd()
data_path = Path('path/to/data')
output_path = PROJECT_ROOT / 'output'
if not output_path.is_dir():
    output_path.mkdir(parents=True)



# %% Pipeline
pipeline = pickle.load(open(str(data_path / "Pipeline.p"), "rb" ) ) #Read the dictionary of the forking paths
avg_mat_corr = pickle.load(open(str(data_path / "Graph_measures" / "corr_with_fisher.p"), "rb" ) ) #Read the correlation matrix of the forking paths
PredictedAcc = pickle.load(open(str(data_path / "predictedVar.pckl"), "rb")) #Read the prediction accuracy of the forking paths, either form the exhaustive search or active learning
n_pipeline = 144
list = []
for n in range (n_pipeline):
    pipe = [str(pipeline['BCT'][n]) + str(pipeline['Weight'][n]) \
              + str(pipeline['Threshold'][n]) + str(pipeline['Negative'][n])
              ,str(PredictedAcc[n])]
    list.append(pipe)

pipeline_name = list

edges = []
for i, var1 in enumerate(pipeline_name):
    for j, var2 in enumerate(pipeline_name):
        if i!= j:
            edges.append((var1, var2, avg_mat_corr[i][j]))

# Create a dataframe from the list of edges
nodes = pd.DataFrame(pipeline_name, columns=['Names', 'Prediction accuracy'])
edges = pd.DataFrame(edges, columns=['source', 'target', 'value'])
nodes.to_csv('D:/ABCD_analyses/Multiverse_analysis/Guardians/Output/nodes.csv', index = False)
edges.to_csv('D:/ABCD_analyses/Multiverse_analysis/Guardians/Output/edges.csv', index = False)
