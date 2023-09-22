"""
This code is am extension of previously proposed method to navigate the results of a multiverse analysis created by Dafflon and colleagues (2022).
The extension implemented in this version is apporaches to handle the combination of brain-wide and region-specific graph measures and to allow
the use of Structural Equation Modeling to infer latent variables.
FOr details, please read the corresponding paper and the original paper proposing the study:
Dafflon et al., “A guided multiverse study of neuroimaging analyses,” Nat. Commun., vol. 13, no. 1, 2022, doi: 10.1038/s41467-022-31347-8.
"""

"""
#1. Setting up environment
##Importing packages 
"""
# %% Import package
from pathlib import Path
import pyreadr
import numpy as np
import random
import pickle
import pandas as pd
import bct
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import copy
import os


"""
##Defining paths
"""
# %% Define paths
PROJECT_ROOT = Path.cwd()
data_path = PROJECT_ROOT / 'data' ##Location of data
output_path = PROJECT_ROOT / 'output' ##Location of output
if not output_path.is_dir():
    output_path.mkdir(parents=True)

warnings.filterwarnings("ignore")


"""
##Setting up the forking paths
"""
# %% Defining variables

def get_data():
    rng = np.random.default_rng(2)
    random.seed(2)
    PROJECT_ROOT = Path.cwd()
    data_path = PROJECT_ROOT / 'data' ##Location of data
    output_path = PROJECT_ROOT / 'output' ##Location of output
    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    ## Graph measures
    graph_meas = pickle.load(open(str(data_path / "Graph_measures.p"), "rb" ) ) #Read the data of graph measures from all forking paths size [Number of subjects,Number of forking paths,Number of brain regions]
    graph_meas = np.transpose(graph_meas, (1,2,0))

    ## Behavior
    Y = pd.read_csv((str(data_path / 'pers.csv'))) #Read the observed variables for SEM


    ## Other related data
    NetworkID = pd.read_csv((str(data_path / '18_Yeo_Network.txt')), sep='\s+', header=None) #Read the network ID for the brain
    NetworkID = NetworkID.iloc[:, 1]
    Pipeline = pickle.load(open(str(data_path / "Pipeline.p"), "rb" ) ) #Read the dictionary of the pipeline

    ## Matching ID
    SubjectID_FC = pd.read_csv((str(data_path / 'subject_list_main_analysis.txt')), sep='\s+', header=None)
    SubjectID_Y = pd.read_csv((str(data_path / 'pers_ID.txt')), sep='\s+', header=None)
    whereID = SubjectID_FC.index.isin(SubjectID_Y.index)
    graph_meas = graph_meas[:,:,whereID]

    n_regions = graph_meas.shape[1]
    n_subject = graph_meas.shape[2]


    ## Partitioning data
    SpaceDefineIdx = 200
    LockBoxDataIdx = 400
    RandomIndexes = rng.choice(SubjectID_Y.shape[0], size=SubjectID_Y.shape[0], replace=False)
    GMModelSpace = graph_meas[:, :, RandomIndexes[0:SpaceDefineIdx]]
    GMLockBoxData = graph_meas[:, :, RandomIndexes[SpaceDefineIdx:LockBoxDataIdx]]
    GMPrediction = graph_meas[:, :, RandomIndexes[LockBoxDataIdx:]]
    YModelSpace = Y.iloc[RandomIndexes[0:SpaceDefineIdx], 1:]
    YLockBoxData = Y.iloc[RandomIndexes[SpaceDefineIdx:LockBoxDataIdx], 1:]
    YPrediction = Y.iloc[RandomIndexes[LockBoxDataIdx:], 1:]

    return NetworkID, n_regions, n_subject, GMModelSpace, GMLockBoxData, GMPrediction, \
           YModelSpace, YLockBoxData, YPrediction, Pipeline
    
NetworkID, n_regions, n_subject, GMModelSpace, GMLockBoxData, GMPrediction, \
           YModelSpace, YLockBoxData, YPrediction, pipeline = get_data()


"""
#2. Building the space 
#This step will compute the the correlation of of the pipelines (bottom-left matrix of Fig. 1).
Here is the first extension of the original study allowing the combination of both brain-wide and region-specific
graph measures.
"""

# %% Computing graph measures

GM = GMModelSpace
n_subject = GM.shape[2]
n_regions = GM.shape[1]

mat_corr = np.zeros((419, 144, 144))
for a in range(419):
    for p1 in range(144):
        for p2 in range (144):
            r = np.corrcoef(GM[p1, a, :], GM[p2, a, :])[0,1]
            mat_corr[a, p1, p2] = r

def fisher_transform(matrix):
    return 0.5 * np.log((1 + matrix) / (1 - matrix))

def reverse_fisher_transform(matrix):
    return (np.exp(2 * matrix) - 1) / (np.exp(2 * matrix) + 1)

transformed_mat = fisher_transform(mat_corr)
average_transformed_matrix = np.mean(transformed_mat, axis=0)
avg_mat_corr = reverse_fisher_transform(average_transformed_matrix)


pickle.dump( avg_mat_corr, open(str(output_path / "graph_corr_with_fisher.p"), "wb" ) )


"""
#3. Space reduction
This step reduce the diemsnion of the similarity matrix fomr (Number of forking paths, Number of forking paths) to (Number of forking paths, 2)
"""
# %% reading the space data
from sklearn import manifold, datasets
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from functools import partial
from time import time
import pickle

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

from umap.umap_ import UMAP
import phate
from sklearn.decomposition import PCA

# Load the previous results
Results = pickle.load(open(str(output_path / "graph_corr_with_fisher.p"), "rb" ) ) # Read the similarity matrix
BCT_Run = pipeline['BCT']
weight_Run = pipeline['Weight']
threshold_Run = pipeline['Threshold']
negative_Run = pipeline['Negative']
preprocessing = [ "abs", "keep", "zero"]



#Scale the data prior to dimensionality reduction

scaler = StandardScaler()
X = scaler.fit_transform(Results.T)
X = X.T
n_neighbors = 20
n_components = 2 #number of components requested. In this case for a 2D space.

#Define different dimensionality reduction techniques 
methods = OrderedDict()
LLE = partial(manifold.LocallyLinearEmbedding,
              n_neighbors = n_neighbors, n_components = n_components, eigen_solver='dense')
methods['LLE'] = LLE(method='standard', random_state=0)
methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                           n_neighbors=n_neighbors, random_state=0)
methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                 random_state=0)
methods['UMAP'] = UMAP(random_state=40, n_components=2, n_neighbors=200,
                             min_dist=.8)

methods['PHATE'] = phate.PHATE()
methods['PCA'] = PCA(n_components=2)

markers      = ["v", "s", "o", "*", "D", "1", "x", "H"]                          # Graph Measures
colourmaps   = {"weighted": "YlGn", "binarize": "Purples"}                    # Weighted and binarize
hatches      = {"abs": "--", "keep": "||", "zero": "**"}                         # Negative values

BCT = np.array(list(BCT_Run.items()))[:,1]
Weight = np.array(list(weight_Run.items()))[:,1]
Threshold = np.array(list(threshold_Run.items()))[:,1]
Negative = np.array(list(negative_Run.items()))[:,1]
BCT_list = list(np.unique(BCT))


# Reduced dimensions
data_reduced = {}

gsDE, axs = plt.subplots(3,2, figsize=(16,16), constrained_layout=True)
axs = axs.ravel()

#Perform embedding and plot the results (including info about the approach in the color/intensity and shape).


for idx_method, (label, method) in enumerate(methods.items()):
    Y = method.fit_transform(X)
    # Save the results
    data_reduced[label] = Y
    Lines={}
    HatchPatch = {}
    for id_neg, neg in enumerate(preprocessing):
        BCTTemp=BCT[Negative==neg]
        weightTemp = Weight[Negative==neg]
        thresholdTemp = Threshold[Negative==neg]
        YTemp=Y[Negative==neg,:]
        
        for wei in (Weight):
            for idx_bct, bct_model in enumerate(BCT_list):
                mask = (BCTTemp == bct_model) & (weightTemp == wei)
                x_values = YTemp[:, 0][mask]
                y_values = YTemp[:, 1][mask]
                threshold_values = thresholdTemp[mask]
                hatch=hatches[neg]

                norm = mcolors.Normalize(vmin=np.min(threshold_values), vmax=np.max(threshold_values))
                scatter = axs[idx_method].scatter(x_values, y_values, c=threshold_values, marker=markers[idx_bct],
                                    hatch=hatches[neg], norm=norm, cmap=colourmaps[wei], s=80)
                Lines[idx_bct] = (mlines.Line2D([], [], color='black', linestyle='None',
                                                marker=markers[idx_bct],  markersize=10, label=bct_model))
        HatchPatch[id_neg] = mpatches.Patch(facecolor=[0.1, 0.1, 0.1],
                                                hatch = hatches[neg],
                                                label = neg,
                                                alpha = 0.1)

# For visualisation purposes show the y and x labels only ons specific plots
    if idx_method % 2 == 0: 
        axs[idx_method].set_ylabel('Dimension 1',fontsize=20)
    if (idx_method == 4) or (idx_method == 5):
        axs[idx_method].set_xlabel('Dimension 2',fontsize=20)

    axs[idx_method].set_title("%s " % (label),fontsize=20, fontweight="bold")
    axs[idx_method].axis('tight')
    axs[idx_method].tick_params(labelsize=15)

OrangePatch = mpatches.Patch(color='orange', label='Weighted network')
PurplePatch = mpatches.Patch(color=[85 / 255, 3 / 255, 152 / 255], label='Binarized network')

IntensityPatch1 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.5', alpha=1)
IntensityPatch2 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.3', alpha=0.4)
IntensityPatch3 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.1', alpha=0.1)

BlankLine = mlines.Line2D([], [], linestyle='None')

"""
gsDE.legend(handles=[HatchPatch[0], HatchPatch[1], HatchPatch[2], BlankLine,
                       OrangePatch, PurplePatch,BlankLine,IntensityPatch1,
                       IntensityPatch2, IntensityPatch3,BlankLine, 
                       Lines[0],Lines[1],Lines[2],Lines[3],Lines[4],Lines[5],
                       Lines[6], Lines[7]],fontsize=15,
                       frameon=False,bbox_to_anchor=(1.05, 0.7))

"""
#save plots locally
gsDE.savefig(str(output_path / 'DifferentEmbeddings_with_Fisher.png'), dpi=300, bbox_inches='tight')
gsDE.savefig(str(output_path / 'DifferentEmbeddings_with_Fisher.svg'), format="svg", bbox_inches='tight')
gsDE.show()


#Do the same as above but for MDS
# %% MDS approach

methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=10, 
                              random_state=21, metric=True)

Y = methods['MDS'].fit_transform(X)
data_reduced['MDS'] = Y

pickle.dump( data_reduced, open(str(output_path / "embeddings.p"), "wb" ) )

Y = data_reduced['MDS']
figMDS = plt.figure(constrained_layout=False, figsize=(21, 15))
gsMDS = figMDS.add_gridspec(nrows=15, ncols=20)
axs = figMDS.add_subplot(gsMDS[:, 0:15])
axs.spines['top'].set_linewidth(1.5)
axs.spines['right'].set_linewidth(1.5)
axs.spines['bottom'].set_linewidth(1.5)
axs.spines['left'].set_linewidth(1.5)
axs.set_xlabel('Dimension 2', fontsize=20, fontweight="bold")
axs.set_ylabel('Dimension 1', fontsize=20, fontweight="bold")
axs.tick_params(labelsize=15)
axs.set_title('Multi-dimensional Scaling', fontsize=25, fontweight="bold")

legend_handles = []

# Create an empty list to store legend handles for bct_model markers
marker_legend_handles = []
HatchPatch = {}
for id_neg, neg in enumerate(preprocessing):
    BCTTemp = BCT[Negative == neg]
    weightTemp = Weight[Negative == neg]
    thresholdTemp = Threshold[Negative == neg]
    YTemp = Y[Negative == neg, :]
    Lines={}
    
    for wei in Weight:
        for idx_bct, bct_model in enumerate(BCT_list):
            mask = (BCTTemp == bct_model) & (weightTemp == wei)
            x_values = YTemp[:, 0][mask]
            y_values = YTemp[:, 1][mask]
            threshold_values = thresholdTemp[mask]
            hatch=hatches[neg]

            norm = mcolors.Normalize(vmin=np.min(threshold_values), vmax=np.max(threshold_values))
            scatter = axs.scatter(x_values, y_values, c=threshold_values, marker=markers[idx_bct],
                                   hatch=hatches[neg], norm=norm, cmap=colourmaps[wei], s=80)

            # Create a legend handle for the bct_model marker and add it to marker_legend_handles
            Lines[idx_bct] = (mlines.Line2D([], [], color='black', linestyle='None',
                                                marker=markers[idx_bct],  markersize=10, label=bct_model))
    HatchPatch[id_neg] = mpatches.Patch(facecolor=[0.1, 0.1, 0.1],
                                                hatch = hatches[neg],
                                                label = neg,
                                                alpha = 0.1)

# Define legend patches here (outside of the loop)
OrangePatch = mpatches.Patch(color='yellow', label='Weighted network')
PurplePatch = mpatches.Patch(color=[85 / 255, 3 / 255, 152 / 255], label='Binarized network')

IntensityPatch1 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.5', alpha=1)
IntensityPatch2 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.3', alpha=0.4)
IntensityPatch3 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='threshold: 0.1', alpha=0.1)

BlankLine = mlines.Line2D([], [], linestyle='None')


# Create a legend for the other elements
figMDS.legend(handles=[HatchPatch[0], HatchPatch[1], HatchPatch[2], BlankLine,
                       OrangePatch, PurplePatch,BlankLine,IntensityPatch1,
                       IntensityPatch2, IntensityPatch3,BlankLine, 
                       Lines[0],Lines[1],Lines[2],Lines[3],Lines[4],Lines[5],
                       Lines[6], Lines[7]],fontsize=15,
                       frameon=False,bbox_to_anchor=(1.05, 0.7))


 
figMDS.savefig(str(output_path / 'MDSSpace_with_fisher.png'), dpi=300)
figMDS.savefig(str(output_path /'MDSSpace_with_fisher.svg'), format="svg")


# Save results form the embedding to be used in the remaining analysis

pickle.dump(data_reduced, open(str(output_path / "embeddings_with_fisher.p"), "wb" ) )

# %% Analyse the neighbours
data_reduced = pickle.load(open(str(output_path / "embeddings_with_fisher.p"), "rb" ) )

from helper import (get_models_neighbours, get_dissimilarity_n_neighbours,
                            get_null_distribution)

N = 143
n_neighbors_step = 10

neighbours_orig, adj_array = get_models_neighbours(N, n_neighbors_step, X)

neighbours_tsne, _ = get_models_neighbours(N, n_neighbors_step,
                                           data_reduced['t-SNE'])
diss_tsne = get_dissimilarity_n_neighbours(neighbours_orig, neighbours_tsne)
del neighbours_tsne

neighbours_lle, _ = get_models_neighbours(N, n_neighbors_step, 
                                          data_reduced['LLE'])
diss_lle = get_dissimilarity_n_neighbours(neighbours_orig,neighbours_lle)
del neighbours_lle

neighbours_se, _ = get_models_neighbours(N, n_neighbors_step,
                                         data_reduced['SE'])
diss_se = get_dissimilarity_n_neighbours(neighbours_orig,neighbours_se)
del neighbours_se

neighbours_mds, _ = get_models_neighbours(N, n_neighbors_step,
                                          data_reduced['MDS'])
diss_mds = get_dissimilarity_n_neighbours(neighbours_orig,neighbours_mds)
del neighbours_mds

neighbours_pca, _ = get_models_neighbours(N, n_neighbors_step,
                                          data_reduced['PCA'])
diss_pca = get_dissimilarity_n_neighbours(neighbours_orig, neighbours_pca)
del neighbours_pca

null_distribution = get_null_distribution(N, n_neighbors_step)

fig, ax = plt.subplots(figsize=(8, 6))
n_neighbours = range(2, N, n_neighbors_step)
ax.plot(n_neighbours, diss_tsne, label='t-SNE', color='#1DACE8')
ax.plot(n_neighbours, diss_lle, label='LLE', color='#E5C4A1')
ax.plot(n_neighbours, diss_se, label='SE', color='#F24D29')
ax.plot(n_neighbours, diss_mds, label='MDS', color='#1C366B')
ax.plot(n_neighbours, diss_pca, label='PCA', color='r')
plt.plot(n_neighbours, null_distribution, label='random', c='grey')
plt.ylim([0,1])
plt.xlim([0,N])
plt.legend(frameon=False)
plt.xlabel('$k$ Nearest Neighbors')
plt.ylabel('Dissimilarity $\epsilon_k$')
plt.savefig(str(output_path / 'dissimilarity_all_with_Fisher.svg'))
plt.savefig(str(output_path / 'dissimilarity_all_with_Fisher.png'), dpi=300)
plt.show()


# %%
"""
#4. Exhaustive Search
This step manually explore all the forking paths to obtain "true prediction performance"
"""
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import warnings
import semopy
warnings.filterwarnings("ignore")

from helper import sem_regression

NetworkID, n_regions, n_subject, GMModelSpace, GMLockBoxData, GMPrediction, \
           YModelSpace, YLockBoxData, YPrediction, pipeline = get_data()

BCT_Run = pipeline['BCT']
weight_Run = pipeline['Weight']
threshold_Run = pipeline['Threshold']
negative_Run = pipeline['Negative']

# Load embedding results. This cell is only necessary if you are running this
# part of the analysis separatly.
ModelEmbeddings = pickle.load(open(str(output_path / "embeddings.p"), "rb" ) )
ModelEmbedding = ModelEmbeddings['MDS']

PredictedVar = np.zeros((len(negative_Run)))
YPred = YPrediction


for i in tqdm(range(len(negative_Run))):
    TempGM = pipeline["BCT"][i]
    tempPredVar = sem_regression(i, YPred, GMPrediction, output_guar, TempGM, AL=0)
    PredictedVar[i] = tempPredVar

#Display how predicted accuracy is distributed across the low-dimensional space
plt.scatter(ModelEmbedding[0: PredictedVar.shape[0], 0],
            ModelEmbedding[0: PredictedVar.shape[0], 1],
            c=PredictedVar, cmap='viridis')
plt.colorbar()
pickle.dump(PredictedVar, open(str(output_path / 'predictedVar.pckl'), 'wb'))


"""
#5. Active Learning
This step performs the active learning
"""


from itertools import product
import pickle

from matplotlib import cm
import bct
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVR
from sklearn.model_selection import permutation_test_score
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from helper import (initialize_bo, run_bo, posterior, 
                             posteriorOnlyModels, display_gp_mean_uncertainty,
                             plot_bo_estimated_space, plot_bo_evolution, plot_bo_repetions)

NetworkID, n_regions, n_subject, GMModelSpace, GMLockBoxData, GMPrediction, \
           YModelSpace, YLockBoxData, YPrediction, pipeline = get_data()

ModelEmbeddings = pickle.load(open(str(output_path / "embeddings_with_Fisher.p"), "rb" ))
ModelEmbedding = ModelEmbeddings['MDS']

PredictedAcc = pickle.load(open(str(output_path / "predictedVar.pckl"), "rb"))

BCT_Run = pipeline['BCT']
weight_Run = pipeline['Weight']
threshold_Run = pipeline['Threshold']
negative_Run = pipeline['Negative']

model_config = {}
model_config['negative_Run'] = negative_Run
model_config['threshold_Run'] = threshold_Run
model_config['weight_Run'] = weight_Run
model_config['BCT_Run'] = BCT_Run
model_config['CommunityIDs'] = NetworkID
model_config['GMPrediction'] = GMPrediction
model_config['GMLockBoxData'] = GMLockBoxData
AL = 1


## Exploratory analysis

##Note: This step takes about 30min.

YPred = YPrediction
kappa = 10

# Define settings for the analysis
kernel, optimizer, utility, init_points, n_iter, pbounds, nbrs, RandomSeed = \
                      initialize_bo(ModelEmbedding, kappa)

# Perform optimization. Given that the space is continuous and the analysis 
# approaches are not, we penalize suggestions that are far from any actual 
# analysis approaches. For these suggestions the registered value is set to the
#  lowest value from the burn in. These points (BadIters) are only used
# during search but exluded when recalculating the GP regression after search.
BadIter = run_bo(optimizer, utility, init_points,
                 n_iter, pbounds, nbrs, RandomSeed,
                 ModelEmbedding, model_config, 
                 YPred,
                 AL, 
                 output_path, BCT_Run,
                 verbose=False)

x_exploratory, y_exploratory, z_exploratory, x, y, gp, vmax, vmin = \
                                           plot_bo_estimated_space(kappa, BadIter,
                                              optimizer, pbounds, 
                                              ModelEmbedding, PredictedAcc, 
                                              kernel, output_path)

# Display the results of the active search and the evolution of the search
# after 5, 10,20, 30 and 50 iterations.
corr = plot_bo_evolution(kappa, x_exploratory, y_exploratory, z_exploratory, x, y, gp,
                  vmax, vmin, ModelEmbedding, PredictedAcc, output_path)

print(f'Spearman correlation {corr}')


# %%
## Exploitatory analysis

kappa = .1

# Define settins for the analysis
kernel, optimizer, utility, init_points, n_iter, pbounds, nbrs, RandomSeed = \
                      initialize_bo(ModelEmbedding, kappa)

# Perform optimization. Given that the space is continuous and the analysis 
# approaches are not, we penalize suggestions that are far from any actual 
# analysis approaches. For these suggestions the registered value is set to the
#  lowest value from the burn in. These points (BadIters) are only used
# during search but exluded when recalculating the GP regression after search.
BadIter = run_bo(optimizer, utility, init_points,
                 n_iter, pbounds, nbrs, RandomSeed,
                 ModelEmbedding, model_config, 
                 YPred,
                 AL, 
                 output_path,
                 verbose=False)

x_exploratory, y_exploratory, z_exploratory, x, y, gp, vmax, vmin = \
                                           plot_bo_estimated_space(kappa, BadIter,
                                              optimizer, pbounds, 
                                              ModelEmbedding, PredictedAcc, 
                                              kernel, output_path)

# Display the results of the active search and the evolution of the search
# after 5, 10,20, 30 and 50 iterations.
corr = plot_bo_evolution(kappa, x_exploratory, y_exploratory, z_exploratory, x, y, gp,
                  vmax, vmin, ModelEmbedding, PredictedAcc, output_path)

print(f'Spearman correlation {corr}')


# %%


"""
#6. Iteration
This steps performs repetitions to test the robustness of the active learning 
"""

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import PredefinedSplit


n_repetitions = 20
kappa = 10
AL = 2
BCT_Run = pipeline['BCT']

BestModelGPSpace = np.zeros(n_repetitions)
BestModelGPSpaceModIndex = np.zeros(n_repetitions)
BestModelEmpirical = np.zeros(n_repetitions)
BestModelEmpiricalModIndex = np.zeros(n_repetitions)
ModelActualAccuracyCorrelation = np.zeros(n_repetitions)
CVPValBestModels = np.zeros(n_repetitions)
PredictedVarRep = np.zeros(n_repetitions)
#predictions = np.zeros((n_repetitions, len(AgesLockBoxData)))

for DiffInit in range(n_repetitions):
    # Define settings for the analysis
    kernel, optimizer, utility, init_points, n_iter, pbounds, nbrs, RandomSeed = \
                      initialize_bo(ModelEmbedding, kappa, repetitions=True,
                                    DiffInit=DiffInit)
    
    # Run BO on the Prediction again
    FailedIters = run_bo(optimizer, utility, init_points,
                        n_iter, pbounds, nbrs, RandomSeed,
                        ModelEmbedding, model_config, 
                        YPred,
                        AL,
                        output_guar, BCT_Run,
                        verbose=False)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=10)

    x_temp = np.array([[res["params"]["b1"]] for res in optimizer.res])
    y_temp = np.array([[res["params"]["b2"]] for res in optimizer.res])
    z_temp = np.array([res["target"] for res in optimizer.res])

    x_obs = x_temp[FailedIters==0]
    y_obs = y_temp[FailedIters==0]
    z_obs = z_temp[FailedIters==0]
    
    muModEmb, sigmaModEmb, gpModEmb = posteriorOnlyModels(gp, x_obs, y_obs, z_obs,
                                                      ModelEmbedding)

    BestModelGPSpace[DiffInit] = muModEmb.max()
    BestModelGPSpaceModIndex[DiffInit] = muModEmb.argmax()
    BestModelEmpirical[DiffInit] = z_obs.max()
    Model_coord = np.array([[x_obs[z_obs.argmax()][-1], y_obs[z_obs.argmax()][-1]]])
    BestModelEmpiricalModIndex[DiffInit] = nbrs.kneighbors(Model_coord)[1][0][0]
    ModelActualAccuracyCorrelation[DiffInit] = spearmanr(muModEmb, PredictedAcc)[0]
    
    TempModelNum = muModEmb.argmax()

    TempResultsLockData = GMLockBoxData[TempModelNum,:,:]
    TempPredictionsData = GMPrediction[TempModelNum,:,:]
    TempResultsLockData = np.transpose(TempResultsLockData, (1,0))
    TempPredictionsData = np.transpose(TempPredictionsData, (1,0))
    TempGM = pipeline["BCT"][TempModelNum]
    # Load the Lockbox data
    tempPredVar = sem_regression(TempModelNum, YLockBoxData, GMLockBoxData, output_guar, TempGM, AL = 2)
    PredictedVarRep[DiffInit] = tempPredVar

plot_bo_repetions(ModelEmbedding, PredictedAcc, BestModelGPSpaceModIndex,
                  BestModelEmpiricalModIndex, BestModelEmpirical,
                  ModelActualAccuracyCorrelation, output_guar)


import pandas as pd
# Obtain the list of 20 models that were defined as the best models
df = pd.DataFrame({'Negative': negative_Run,'Threshold': threshold_Run, 
                   'weight': weight_Run, 'bct': BCT_Run})
df_best = df.iloc[BestModelEmpiricalModIndex]
df_best['var']= PredictedVarRep
df_best

repetions_results = {
                      'dataframe': df_best,
                      'BestModelGPSpaceModIndex': BestModelGPSpaceModIndex,
                      'BestModelEmpiricalIndex':  BestModelEmpiricalModIndex, 
                      'BestModelEmpirical': BestModelEmpirical,
                      'ModelActualAccuracyCorrelation': ModelActualAccuracyCorrelation
                     }
pickle.dump( repetions_results, open(str(output_guar / "repetitions_results_with_Fisher.p"), "wb" ) )

print(df_best)
