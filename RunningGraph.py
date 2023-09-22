"""
This code is used for generating graph measures from all the forking paths.
"""

"""
##Setting the environment
"""
# %% Import package
from pathlib import Path
import pyreadr
import numpy as np
import random
import pickle
import pandas as pd
from copy import deepcopy

"""
##Defining paths
"""
# %% Define paths
PROJECT_ROOT = Path.cwd()
data_path = Path('Path/to/data')
output_path = PROJECT_ROOT / 'output'
if not output_path.is_dir():
    output_path.mkdir(parents=True)


"""
##Setting up the forking paths
"""
# %% Defining variables
import bct
import warnings
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
def define_variables(p = False):
    rng = np.random.default_rng(2)
    random.seed(2)

    #Define path
    PROJECT_ROOT = Path.cwd()
    data_path = Path('Path/to/data')
    output_path = PROJECT_ROOT / 'output'
    if not output_path.is_dir():
        output_path.mkdir(parents=True)

        #Load data
    FC_main = pyreadr.read_r(data_path / 'FC_mat.rds') #Read the FC data, size (number of subjects, number of regions, number of regions)
    FC_main = (FC_main[None])

        
    NetworkID = pd.read_csv((str(data_path / '18_Yeo_Network.txt')), sep='\s+', header=None)
    NetworkID = NetworkID.iloc[:, 1]
    n_regions = FC_main.shape[1]
    n_subject = FC_main.shape[0]
    FC = np.asarray(FC_main)
    FC = np.nan_to_num(FC,copy=True,nan=1.0)
    FC = np.transpose(FC, (1,2,0))
    

    return NetworkID, n_regions, n_subject, FC
    

def analysis_space(tempBCT, BCT_models, x, NetworkID, weight):
    ###This function is used to call the corresponding package and function to calculate graph measures
    if tempBCT == 'local efficiency':
        if weight == True:
            ss = BCT_models[tempBCT](x,1)
        else:
            ss = bct.efficiency_bin(x,1)
    elif tempBCT == 'global efficiency':
        if weight == True:
            ss = BCT_models[tempBCT](x,0)
        else:
            ss = bct.efficiency_bin(x,0)
    elif tempBCT == 'modularity':
        _, ss = BCT_models[tempBCT](x, seed=2)
    elif tempBCT == 'eigenvector centrality':
        ss = BCT_models[tempBCT](x)
    elif tempBCT == 'participation coefficient':
        if weight == True:
            temp = BCT_models[tempBCT](x, NetworkID)
            ss = temp[0]
        else:
            temp = bct.participation_coef(x,NetworkID)
            ss = temp[0]
    elif tempBCT == 'betweennness centrality':
        if weight == True:
            ss = BCT_models[tempBCT](x)
        else:
            ss = bct.betweenness_bin(x)
    elif tempBCT == 'strength':
        if weight == True:
            ss = BCT_models[tempBCT](x)
        else:
            ss = bct.degrees_und(x)
    else:
        if weight == True:
            ss = BCT_models[tempBCT](x)
        else:
            ss = bct.clustering_coef_bu(x)
    return ss


def define_pipelines():
    #This function defines the forking paths
    #Forking paths of teh graph measures
    BCT_models = {
    'strength': bct.strengths_und,  #Output vector for each node
    'betweennness centrality': bct.betweenness_wei, #output vector of each node, strength has to be converted to length
    'clustering': bct.clustering_coef_wu, #output vector for each node
    'eigenvector centrality': bct.eigenvector_centrality_und,   #for bin and wei, output vector for each node
    'local efficiency' : bct.efficiency_wei,  #local = T, the weight is transferred to length within the function, 
                                                #only positive weights, output vector for each node
    'global efficiency' : bct.efficiency_wei,
    'modularity': bct.modularity_louvain_und_sign,   #for bin and wei, output vector for node assignment and one value for modularity
    'participation coefficient': bct.participation_coef_sign,   #Output has 2, positive and negative weights
    #'shortest path length': bct. 
    }

    weight_options   = ["weighted", "binarize"] #Forking paths of the edges
    thresholds       = [0.5, 0.3, 0.1]  #Forking paths of graph threshold
    neg_options      = [ "abs", "keep", "zero"] #Forking paths of how to handle negative values in FC

    BCT_Run = {}
    weight_Run = {}
    threshold_Run = {}
    negative_Run = {}
    count = 0
    
    for neg_idx, tempNeg in enumerate (neg_options):
        for thr_idx, tempThr in enumerate (thresholds):
            for weight_idx, tempWeight in enumerate (weight_options):
                for tempBCT in BCT_models.keys():
                    BCT_Run[count] = tempBCT
                    weight_Run[count] = tempWeight
                    threshold_Run[count] = tempThr
                    negative_Run[count] = tempNeg
                    count += 1
    PipelineResults={
                "BCT": BCT_Run,
                "Weight": weight_Run,
                "Threshold": threshold_Run,
                "Negative": negative_Run}
    
    return PipelineResults, BCT_models

def neg_corr(option, f):
    #This function specifies how to handle negative values
    if  option == "abs":
        out = abs(f)
    elif option == "zero": 
        out = deepcopy(f)
        out[np.where(f < 0)] = 0
    elif option == "keep":
        out = deepcopy(f)
    return out

def compute_measures(FCsub, Pipeline, BCT_models):
    n_regions = FC.shape[0]
    n_pipe = len(Pipeline["Weight"])
    allMeasures = np.zeros((n_pipe,n_regions))
    for idx_pipe in range(n_pipe):
        tempNeg = Pipeline["Negative"][idx_pipe]
        tempThr = Pipeline["Threshold"][idx_pipe]
        tempWei = Pipeline["Weight"][idx_pipe]
        tempBCT = Pipeline["BCT"][idx_pipe]

        FCin = neg_corr(tempNeg, FCsub)
        tempFC = bct.threshold_proportional(FCin, tempThr, copy = True)
        if tempWei == "weighted":
            if tempBCT == 'betweennness centrality':
                x = bct.weight_conversion(tempFC, 'lengths', copy = True)
            else:
                x = bct.weight_conversion(tempFC, 'normalize', copy = True)
            graph_measures = analysis_space(tempBCT, BCT_models, x, NetworkID, weight = True)

        else:
            x = bct.weight_conversion(tempFC, 'binarize', copy = True)
            graph_measures = analysis_space(tempBCT, BCT_models, x, NetworkID, weight = False)
        
        if graph_measures.size == 1: graph_measures = [graph_measures] * n_regions
        elif graph_measures.size == 2: 
            graph_measures = [[graph_measures[0]] * int(n_regions/2), [graph_measures[1]] * int(n_regions/2)]
            graph_measures = [item for sublist in graph_measures for item in sublist]
        print(tempBCT)
        allMeasures[idx_pipe, :] = graph_measures
    return allMeasures

def run_subjects(i):
    FCsub = FC[:,:,i]
    measureResult = compute_measures(FCsub, Pipeline, BCT_models)
    print(i)
    return measureResult


NetworkID, n_regions, n_subject, FC = define_variables()
Pipeline, BCT_models = define_pipelines()


allGraphs = {}

for i in range(1):
    results = run_subjects(i)
    allGraphs[i] = results

print(len(allGraphs))


pickle.dump(Pipeline, open(str(output_path / "Pipeline.p"), "wb" ) )
pickle.dump(allGraphs, open(str(output_path / "allGraph.p"), "wb" ) )



