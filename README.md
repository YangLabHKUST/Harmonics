# Harmonics
Hierarchical distribution matching enables comprehensive characterization of common and condition-specific cell niches in spatial omics data.

Benchmarking codes are deposited at [https://zenodo.org/records/16728860](https://zenodo.org/records/16728860).
Preprocessed data are deposited at [https://zenodo.org/records/16794669](https://zenodo.org/records/16794669).
<img width="6976" height="7957" alt="fig1" src="https://github.com/user-attachments/assets/f56bb28c-ef2e-4a11-a043-86f361de1458" />


## System requirements
The package is tested on Linux and Windows


## Installation
### Install from PyPI.
Create an environment.

```
conda create -n Harmonics_CNI python=3.11
conda activate Harmonics_CNI
```

Install Harmonics

```
pip install harmonics-bio
```

### Install from GitHub.
Clone the repository. 

```
git clone https://github.com/YangLabHKUST/Harmonics.git
cd Harmonics
```

Create an environment.

```
conda create -n Harmonics_CNI python=3.11
conda activate Harmonics_CNI
```

Install the required packages.

```
pip install -r requirements.txt
```

Install Harmonics.

```
python setup.py build
python setup.py install
```


## Tutorial
Detailed tutorials covering the usage of Harmonics and the reproduction of experiments can be found at [Read the Docs](https://harmonics-tutorial.readthedocs.io/en/latest/).

## Easy start
Import the package.
```
from Harmonics import *
```

Initialize the model
```
model = Harmonics_Model(adata_list,  # a list of anndata objects, the whole dataset for condition-agnostic dataset / the control group in case-control setting / the reference data for label transfer
                        slice_name_list,  # a list of slice names for corresponding data in adata_list
                        cond_list=cond_list,  # a list of anndata objects, None for condition-agnostic dataset / the case group in case-control setting / the query data for label transfer
                        cond_name_list=cond_name_list,  # a list of slice names for corresponding data in cond_list
                        concat_label='slice_name',  # the key in .obs for storing slice names
                        proportion_label=None,  # the key in .obsm of the cell type deconvolution results for low resolution data
                        refine_k=0,  # niche distribution refinement using 'refine_k' cell types with highest proportion, set to 0 to avoid performing refinement
                        seed=1234,  # random seed
                        parallel=True,  # whether to run in parallel 
                        verbose=True,
                        )
```

Construct cell representations (cell type distribution)
```
model.preprocess(ct_key='celltype',  # cell type key in .obs 
                 spatial_key='spatial',  # spatial coordinate key in .obsm
                 method='joint',  # the mode for graph construction 'joint': 'n_step'-hop delaunay triangulation with graph completion to at least 'n_neighbors' neighbors per cell; 'delaunay': 'n_step'-hop delaunay triangulation; 'knn': 'n_neighbors' neighbors per cell; None: directly use the cell type composition for low resolution data
                 n_step=3,  
                 n_neighbors=20,
                 cut_percentage=99,  # preserving the shortest 'cut_percentage'% edges of the delaunay triangulation adjacency graph
                 )
```

Over-clustering initialization for the whole dataset for condition-agnostic dataset / the control group in case-control setting / the reference data for label transfer
```
model.initialize_clusters(dim_reduction=True,  # default, perform pca or not
                          explained_var=None,  # default, target cumulative explained variance for dimensionality reduction
                          n_components=None,  # default, number of components to retain
                          n_components_max=100,  # default, maximum number of components allowed during reduction
                          standardize=True,  # default, whether to z-score normalize each feature before dimensionality reduction
                          method='kmeans',  # default, method for initialization
                          Qmax=20,  # default, number of clusters for initialization
                          )
```

Perform HDM to find solution
```
model.hier_dist_match(assign_metric='jsd',  # metric for evaluting distribution similarity
                      weighted_merge=True,  # set to true to use WJSD in the merging phase
                      max_iters=100,  # max iteration 
                      tol=1e-4,  # the tolerance for convergence
                      Qmin=2,  # the minimun number of niches
                      )
```

Select the solution
```
adata_list, adata_concat = model.select_solution(n_niche=None,  # the number of niches to select, set to None for metric-guided solution selection
                                                 niche_key='niche_label',  # the key in .obs to store the results
                                                 auto=True,  # whether to automatically determine the solution, if 'n_niche=None' than 'auto' should be True
                                                 metric='jsd',  # using the minJSD score for solution selection
                                                 threshold=0.1,  # threshold for minJSD
                                                 return_adata=True,  # whether to return anndata object
                                                 plot=True,  # whether to plot the minJSD curve
                                                 save=False,  # whether to save the minJSD curve
                                                 fig_size=(10, 6),  # figure size
                                                 save_dir=None,  # save path
                                                 file_name=f'score_vs_nichecount_basic.pdf',  # file name
                                                 )
```


Over-clustering initialization for the case group in case-control setting
```
model.initialize_clusters_cond(assign_metric='jsd',  # metric for evaluting distribution similarity when assigning cells to BCNs
                               threshold=0.1,  # min divergence below this threshold will be assigned to BCNs 
                               min_cell_per_niche=100,  # the least average cell amount for each niche
                               dim_reduction=True,  # default, perform pca or not
                               explained_var=None,  # default, target cumulative explained variance for dimensionality reduction
                               n_components=None,  # default, number of components to retain
                               n_components_max=100,  # default, maximum number of components allowed during reduction
                               standardize=True,  # default, whether to z-score normalize each feature before dimensionality reduction 
                               method='kmeans',  # default, method for initialization 
                               Rmax=10,  # default, number of clusters for initialization 
                               )
```

Perform HDM to find solution
```
model.hier_dist_match_cond(assign_metric='jsd',  # metric for evaluting distribution similarity
                           weighted_merge=True,  # set to true to use WJSD in the merging phase
                           max_iters=100,  # max iteration 
                           tol=1e-4,  # the tolerance for convergence 
                           )
```


Select the solution
```
cond_list, cond_concat = model.select_solution_cond(n_csn=None,  # the number of CSNs to select, set to None for metric-guided solution selection
                                                    niche_key='niche_label',  # the key in .obs to store the results niche assignment result
                                                    csn_key='csn_label',  # the key in .obs to store the results CSN assignment result
                                                    auto=True,  # whether to automatically determine the solution, if 'n_csn=None' than 'auto' should be True
                                                    metric='jsd',  # using the minJSD score for solution selection
                                                    threshold=0.1,  # threshold for minJSD
                                                    return_adata=True,  # whether to return anndata object
                                                    plot=True,  # whether to plot the minJSD curve
                                                    save=False,  # whether to save the minJSD curve
                                                    fig_size=(10, 6),  # figure size
                                                    save_dir=None,  # save path 
                                                    file_name='score_vs_nichecount_cond.pdf',  # file name
                                                    )
```

Label transfer
```
trans_list, trans_concat = model.label_transfer(assign_metric='jsd',  # metric for evaluting distribution similarity when assigning cells to niches
                                                niche_key='niche_label',  # the key in .obs to store the results niche assignment result 
                                                return_adata=True,  # whether to return anndata object
                                                )
```
