##########################################################
# SPiSCy supplementary data figure S4                    #
# Author: Emilie Roy                                     #
# Run clustering algorithms 3 times and measure runtime  #
# Run with 100k and 1M cells                             #
# PCA for all methods except CytoVI (VAE)                #
##########################################################

import sys
from multiprocessing import freeze_support
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import os
from datetime import datetime
import flowsom as fs
from sklearn.cluster import Birch
from scvi.external import cytovi
from sklearn.cluster import HDBSCAN
import parc
import phenograph as ph
from sklearn.decomposition import PCA


# time functions
def start_time():
    """
    Get script start time
    Returns start_time as a datetime.now() format
    """
    start_time = datetime.now()
    print("Start:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    return start_time

def end_time():
    """
    Get script end time
    Returns end_time as a datetime.now() format
    """
    end_time = datetime.now()
    print("End:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    return end_time

def elapsed_time(start_time, end_time):
    """
    Calculate total script execution time. Prints execution time without miliseconds
    
    :param start_time: datetime format
    :param end_time: datetime format
    """
    elapsed = end_time - start_time
    elapsed_short = str(elapsed).split(".")[0]
    return elapsed


# function random sample n

def downsample_pd_dataframe(dataframe, sample_size):
    """
    Randomly samples a total of `sample_size` rows from the entire dataframe,
    regardless of filename.
    
    :param dataframe: pandas dataframe with a 'filename' column
    :param sample_size: total number of rows to sample
    """
    total_cells = len(dataframe)
    n_to_sample = min(sample_size, total_cells)
    
    print(f"Sampling a total of {n_to_sample} cells from {len(dataframe['filename'].unique())} files")
    
    small_df = dataframe.sample(n=n_to_sample, random_state=17).reset_index(drop=True)
    
    print(f"Total number of cells after sampling: {small_df.shape[0]}\n")
    
    return small_df


# function PCA

def fit_transform_PCA(input_df, markers_to_reduce=None, n_components=None):
    """
    Fit a dimensionality reduction method of choice on dataframe and transform the data
    Method optionsa are: directly use the markers values, PCA, KernelPCA, Isomap, or FastICA
    
    :param method_choice: string from dim_reduction.yaml config file indicating chosen method
    :param input_df: pandas dataframe that must be transformed
    :rest: parameters for each method from dim_reduction.yaml

    Returns transformed data as pandas dataframe, names of columns, and the fitted reducer
    """

    print(f"Dimensionality reduction via PCA on {markers_to_reduce}. n_components={n_components}", flush=True)
    reducer = PCA(n_components=n_components, random_state=17)
    reduced_columns = [f"PC{x}" for x in range(1, n_components + 1)]

    index = input_df.index
    input_dr = input_df[markers_to_reduce].to_numpy(dtype="float32")
    reduced = reducer.fit_transform(input_dr)
    small_reduced_df = pd.DataFrame(reduced, index=index, columns=reduced_columns)

    print(f"Percentage of variance explained by each of the PCA components: {reducer.explained_variance_ratio_}", flush=True)
    
    return small_reduced_df, reduced_columns, reducer



# function prepare anndata for flowsom and

def pd_df_to_anndata(X, df, var_names, sample_key=None):
    columns = ["filename"]

    if sample_key is not None:
        columns.append(sample_key)

    adata = ad.AnnData(X=X,
                    obs=df[columns].copy(),
                    var=pd.DataFrame(index=var_names))
    return adata


# function to run clustering algos (sample, DR, cluster)

def run_flowsom(input_dataframe, sample_size, markers_to_reduce, n_components=4, x_dims=10, y_dims=10, nb_metaclusters=12):
    # random downsample
    small_df = downsample_pd_dataframe(dataframe=input_dataframe, sample_size=sample_size)
    # PCA DR
    small_reduced_df, reduced_columns, reducer = fit_transform_PCA(input_df=small_df,
                                                              markers_to_reduce=markers_to_reduce,
                                                              n_components=n_components)
    # convert to anndata
    adata = pd_df_to_anndata(X=small_reduced_df.values,
                         df=small_df,
                         var_names=reduced_columns)
    # clustering
    start_cluster = start_time()

    fsom = fs.FlowSOM(adata,
                cols_to_use=reduced_columns,
                xdim=x_dims,
                ydim=y_dims,
                n_clusters=nb_metaclusters,
                seed=17)

    end_cluster = end_time()
    cluster_time = elapsed_time(start_cluster, end_cluster)
    print(f"Clustering time = {cluster_time}")
    return cluster_time.total_seconds()

def run_phenograph(input_dataframe, sample_size, markers_to_reduce, n_components=4, k=30, dist_metric="euclidean", community_detection="leiden"):
    # random sample
    small_df = downsample_pd_dataframe(dataframe=input_dataframe, sample_size=sample_size)
    # PCA DR
    small_reduced_df, reduced_columns, reducer = fit_transform_PCA(input_df=small_df,
                                                              markers_to_reduce=markers_to_reduce,
                                                              n_components=n_components)
    # clustering
    start_cluster = start_time()

    communities, graph, Q = ph.cluster(
        small_reduced_df,
        k = k,
        primary_metric = dist_metric,
        clustering_algo = community_detection,
        seed = 17)

    end_cluster = end_time()
    cluster_time = elapsed_time(start_cluster, end_cluster)
    print(f"Clustering time = {cluster_time}")
    return cluster_time.total_seconds()

def run_birch(input_dataframe, sample_size, markers_to_reduce, n_components=4, radius_threshold=0.5, branching_factor=50, nb_clusters=10):
    # random sample
    small_df = downsample_pd_dataframe(dataframe=input_dataframe, sample_size=sample_size)
    # PCA DR
    small_reduced_df, reduced_columns, reducer = fit_transform_PCA(input_df=small_df,
                                                              markers_to_reduce=markers_to_reduce,
                                                              n_components=n_components)
    # clustering
    start_cluster = start_time()

    brc = Birch(threshold=radius_threshold,
                branching_factor=branching_factor,
                n_clusters=nb_clusters,
                compute_labels=True)
    brc.fit(small_reduced_df)

    end_cluster = end_time()
    cluster_time = elapsed_time(start_cluster, end_cluster)
    print(f"Clustering time = {cluster_time}")
    return cluster_time.total_seconds()

def run_parc(input_dataframe, sample_size, markers_to_reduce, n_components=4, jac_weighted_edges=True, jac_std_global=0.2):
    # random sample
    small_df = downsample_pd_dataframe(dataframe=input_dataframe, sample_size=sample_size)
    # PCA DR
    small_reduced_df, reduced_columns, reducer = fit_transform_PCA(input_df=small_df,
                                                              markers_to_reduce=markers_to_reduce,
                                                              n_components=n_components)
    # clustering
    start_cluster = start_time()

    parc1 = parc.PARC(small_reduced_df,
                  random_seed=17,
                  jac_weighted_edges=jac_weighted_edges,
                  jac_std_global=jac_std_global)
    parc1.run_PARC()

    end_cluster = end_time()
    cluster_time = elapsed_time(start_cluster, end_cluster)
    print(f"Clustering time = {cluster_time}")
    return cluster_time.total_seconds()

def run_hdbscan(input_dataframe, sample_size, markers_to_reduce, n_components=4, min_samples=5, min_cluster_size=7):
    # random sample
    small_df = downsample_pd_dataframe(dataframe=input_dataframe, sample_size=sample_size)
    # PCA DR
    small_reduced_df, reduced_columns, reducer = fit_transform_PCA(input_df=small_df,
                                                              markers_to_reduce=markers_to_reduce,
                                                              n_components=n_components)
    # clustering
    start_cluster = start_time()

    hdb = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, copy=True, n_jobs=-1)
    hdb.fit(small_reduced_df)
    
    end_cluster = end_time()
    cluster_time = elapsed_time(start_cluster, end_cluster)
    print(f"Clustering time = {cluster_time}")
    return cluster_time.total_seconds()

def run_cytovi(input_dataframe, sample_size, markers_to_reduce, nb_neighbors_clustering=20):
    # random downsample
    small_df = downsample_pd_dataframe(dataframe=input_dataframe, sample_size=sample_size)
    # DR via direct markers
    small_clustering_df = small_df[markers_to_reduce]
    # convert to anndata
    adata_small = pd_df_to_anndata(X=small_clustering_df.values,
                            df=small_df,
                            var_names=markers_to_reduce)
    # model training
    start_cluster = start_time()

    cytovi.CYTOVI.setup_anndata(adata_small)
    model = cytovi.CYTOVI(adata_small)   
    model.train(n_epochs_kl_warmup=50)
    
    # leiden on latent space
    latent_small = model.get_latent_representation(adata=adata_small, batch_size=1024)
    adata_small.obsm["X_cytovi"] = latent_small
    sc.pp.neighbors(adata_small, n_neighbors=nb_neighbors_clustering, use_rep="X_cytovi", transformer="pynndescent")
    sc.tl.leiden(adata_small, resolution=0.4, key_added="cluster", flavor="igraph")
    
    end_cluster = end_time()
    cluster_time = elapsed_time(start_cluster, end_cluster)
    print(f"Clustering time = {cluster_time}")
    return cluster_time.total_seconds()



# function test an algo (3x runs)

def test_algo(algo, input_dataframe, sample_size, markers_to_reduce, repeats=3):
    times = []

    algo = algo.lower()

    for i in range(1, repeats+1):
        print(f"RUN #{i}", flush=True)
        start = start_time()

        if algo == "flowsom":
            t = run_flowsom(input_dataframe, sample_size, markers_to_reduce)
        elif algo == "parc":
            t = run_parc(input_dataframe, sample_size, markers_to_reduce)
        elif algo == "hdbscan":
            t = run_hdbscan(input_dataframe, sample_size, markers_to_reduce)
        elif algo == "phenograph":
            t = run_phenograph(input_dataframe, sample_size, markers_to_reduce)
        elif algo == "cytovi":
            t = run_cytovi(input_dataframe, sample_size, markers_to_reduce)
        elif algo == "birch":
            t = run_birch(input_dataframe, sample_size, markers_to_reduce)
        else:
            sys.exit("Unknown clustering algorithm")

        end = end_time()
        elapsed_time(start, end)

        times.append(t)

    return times




##################
## MAIN PROGRAM ##
##################

def main():
    # set up inputs
    BASE_DIR = "/mnt/spiscy_sur"
    samples_csv = f"{BASE_DIR}/results/csv/final_samples.csv"
    samples = pd.read_csv(samples_csv)
    markers = ["CD4", "CD8", "CD154", "CD137", "CD25", "CD127", "CD45RA", "CD27", "CD161"]
    sample_sizes = [100_000, 1_000_000]
    algos = ["FlowSOM", "BIRCH", "PARC", "PhenoGraph", "HDBSCAN", "CytoVI"]
    results = {100_000: {}, 1_000_000: {}}

    # run each algo 3x, with both 100k and 1M cells
    for algo in algos:
        for sample_size in sample_sizes:
            print(f"Running algorithm={algo}, sample_size={sample_size}")
            times = test_algo(algo, samples, sample_size, markers)
            results[sample_size][algo] = times
    
    # plotting
    times_100k = results[100_000]
    times_1M = results[1_000_000]

    labels = list(times_100k.keys())
    x = np.arange(len(labels))
    
    # means
    mean_100k = [statistics.mean(v) for v in times_100k.values()]
    mean_1M = [statistics.mean(v) for v in times_1M.values()]

    # std
    std_100k = [statistics.stdev(v) for v in times_100k.values()]
    std_1M = [statistics.stdev(v) for v in times_1M.values()]

    fig, ax = plt.subplots()

    ax.bar(x - 0.2, mean_100k, width=0.4, yerr=std_100k, capsize=5, label="100_000 cells", color="lightgray")
    ax.bar(x + 0.2, mean_1M, width=0.4, yerr=std_1M, capsize=5, label="1_000_000 cells", color="forestgreen", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Time (s)")
    ax.set_title("Runtime comparison")
    ax.set_yscale("log")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()