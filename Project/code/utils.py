import os
import random
import time
import copy
import json
import psutil
import torch
from tqdm.notebook import tqdm
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
import torch.nn.functional as F

from MatrixVectorizer import MatrixVectorizer

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from torch.utils.data import Subset, DataLoader

from MatrixVectorizer import MatrixVectorizer
from models.unet import ContextUnet, ContextUnetGraph

############################## DATA FUNCTIONS ##############################

# loop through data and turn each entry into a matrix (dimxdim)
def vectorize_matrix(df, dim=160):
    """ 
    """
    matrices = []
    for i in range(len(df)):
        vector = df.loc[i].values
        matrices.append(MatrixVectorizer.anti_vectorize(vector, dim, include_diagonal=False))
    return matrices


def split_into_patches(mat, patch_size):
    """
    Splits a 2D matrix into non-overlapping patches.
    
    Args:
        mat: 2D numpy array (e.g., an adjacency matrix).
        patch_size: Size of each patch (assumes square patches).
    
    Returns:
        patches: List of patches (each patch is a 2D numpy array).
        H, W: The number of patches in height and width.
    """
    H_full, W_full = mat.shape
    patches = []
    num_rows = H_full // patch_size
    num_cols = W_full // patch_size
    for i in range(num_rows):
        for j in range(num_cols):
            patch = mat[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)
    return patches, num_rows, num_cols

def reassemble_from_patches(patches, num_rows, num_cols, patch_size):
    """
    Reassembles patches into a full 2D matrix.
    
    Args:
        patches: List of patches (each a 2D numpy array).
        num_rows: Number of patches along height.
        num_cols: Number of patches along width.
        patch_size: Size of each patch.
    
    Returns:
        A 2D numpy array reassembled from patches.
    """
    full = np.zeros((num_rows*patch_size, num_cols*patch_size))
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            full[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[idx]
            idx += 1
    return full


def histogram_matching(source_vals, reference_vals, nbins=256):
    """
    Matches the histogram of source_vals to that of reference_vals.
    
    Both inputs should be flattened 1D arrays.
    """
    source_vals = np.clip(source_vals, 0, 1)
    reference_vals = np.clip(reference_vals, 0, 1)
    
    # Compute histograms
    src_hist, src_bins = np.histogram(source_vals, bins=nbins, range=(0,1), density=True)
    ref_hist, ref_bins = np.histogram(reference_vals, bins=nbins, range=(0,1), density=True)
    
    # Compute CDFs
    src_cdf = np.cumsum(src_hist)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_hist)
    ref_cdf /= ref_cdf[-1]
    
    # Compute bin centers
    src_bin_centers = 0.5 * (src_bins[:-1] + src_bins[1:])
    ref_bin_centers = 0.5 * (ref_bins[:-1] + ref_bins[1:])
    
    # Map the source CDF to the reference values via interpolation.
    mapping = np.interp(src_cdf, ref_cdf, ref_bin_centers)
    
    # Digitize the source values to find the corresponding bin indices.
    src_indices = np.digitize(source_vals, src_bins) - 1
    src_indices = np.clip(src_indices, 0, nbins-1)
    matched = mapping[src_indices]
    return matched

def patch_histogram_matching(lr_mat, hr_mat, patch_size, nbins=256):
    """
    Performs histogram matching patch by patch.
    
    Args:
        lr_mat: LR adjacency matrix as a 2D numpy array.
        hr_mat: HR adjacency matrix as a 2D numpy array or a reference HR matrix.
        patch_size: Size of the patches (e.g., 67 or any factor that divides the matrix size).
        nbins: Number of bins to use in histogram matching.
    
    Returns:
        A new LR matrix where each patch is histogram-matched to the corresponding HR patch.
    """
    lr_patches, num_rows, num_cols = split_into_patches(lr_mat, patch_size)
    hr_patches, _, _ = split_into_patches(hr_mat, patch_size)
    
    matched_patches = []
    for lr_patch, hr_patch in zip(lr_patches, hr_patches):
        # Flatten patches to 1D arrays
        lr_flat = lr_patch.flatten()
        hr_flat = hr_patch.flatten()
        matched_flat = histogram_matching(lr_flat, hr_flat, nbins=nbins)
        matched_patch = matched_flat.reshape(lr_patch.shape)
        matched_patches.append(matched_patch)
        
    matched_lr = reassemble_from_patches(matched_patches, num_rows, num_cols, patch_size)
    return matched_lr

########################### SUBMISSION FUNCTIONS ###########################

def create_kaggle_submission_file(preds, filename='test_submission.csv'):
    """
    Create a submission file for the Kaggle Competition, in the
    format of 'ID,Prediction' for each model predictions.
    Params:
        preds: List of model predictions
        filename: Name of the submission file
    """
    print(f"Creating File at {filename}...")
    with open(filename, 'w') as f:
        f.write('ID,Predicted\n')
        for i, value in enumerate(preds):
            f.write(f'{i+1},{value}\n')
    print("Done!")

############################# METRICS FUNCTIONS ############################

def compute_metrics(fin, real, verbose=True):
    """ 
    Compute evaluation metrics for the model predictions.
    Args:
        fin: List of predicted matrices
        real: List of ground-truth matrices
        verbose: Boolean to print the metrics after computation
    Returns:
        overall_mae: Overall Mean Absolute Error
        pcc: Pearson Correlation Coefficient
        js_dis: Jensen-Shannon Distance
        mae_pc: List of Mean Absolute Error for PageRank Centrality
        mae_ec: List of Mean Absolute Error for Eigenvector Centrality
        mae_bc: List of Mean Absolute Error for Betweenness Centrality
    """
    pred_1d_list, gt_1d_list = [], []
    mae_bc, mae_ec, mae_pc = [], [], []
    all_pred_matrices, all_gt_matrices = [], []

    with torch.no_grad():
        for i in tqdm(range(len(fin))):
            # Post-process predicted matrix: clamp negatives to zero.
            pred_matrix = fin[i]
            pred_matrix[pred_matrix < 0] = 0

            all_pred_matrices.append(pred_matrix)
            all_gt_matrices.append(real[i])

            # Convert adjacency matrices to NetworkX graphs
            pred_graph = nx.from_numpy_array(pred_matrix, create_using=nx.Graph())
            gt_graph = nx.from_numpy_array(real[i], create_using=nx.Graph())

            # Compute centrality measures
            pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
            pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
            pred_pc = nx.pagerank(pred_graph, weight="weight")

            gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
            gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
            gt_pc = nx.pagerank(gt_graph, weight="weight")

            # Convert centrality dictionaries to lists
            pred_bc_values = list(pred_bc.values())
            pred_ec_values = list(pred_ec.values())
            pred_pc_values = list(pred_pc.values())

            gt_bc_values = list(gt_bc.values())
            gt_ec_values = list(gt_ec.values())
            gt_pc_values = list(gt_pc.values())

            # Compute MAEs
            mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
            mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
            mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))

            # Vectorize matrices
            pred_1d_list.append(MatrixVectorizer.vectorize(pred_matrix))
            gt_1d_list.append(MatrixVectorizer.vectorize(real[i]))

    # Concatenate flattened matrices
    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    overall_mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)

    avg_mae_bc = np.mean(mae_bc)
    avg_mae_ec = np.mean(mae_ec)
    avg_mae_pc = np.mean(mae_pc)

    if verbose:
        print("Overall MAE: ", overall_mae)
        print("Pearson Correlation Coefficient (PCC): ", pcc)
        print("Jensen-Shannon Distance: ", js_dis)
        print("Average MAE Betweenness Centrality:", avg_mae_bc)
        print("Average MAE Eigenvector Centrality:", avg_mae_ec)
        print("Average MAE PageRank Centrality:", avg_mae_pc)

    return overall_mae, pcc, js_dis, mae_pc, mae_ec, mae_bc


def fast_metrics(fin, real, verbose=True):
    """ 
    Compute evaluation metrics for the model predictions.
    Args:
        fin: List of predicted matrices
        real: List of ground-truth matrices
        verbose: Boolean to print the metrics after computation
    Returns:
        overall_mae: Overall Mean Absolute Error
        pcc: Pearson Correlation Coefficient
        js_dis: Jensen-Shannon Distance
        mae_pc: List of Mean Absolute Error for PageRank Centrality
        mae_ec: List of Mean Absolute Error for Eigenvector Centrality
        mae_bc: List of Mean Absolute Error for Betweenness Centrality
    """
    pred_1d_list, gt_1d_list = [], []
    mae_bc, mae_ec, mae_pc = [], [], []

    with torch.no_grad():
        for i in tqdm(range(len(fin))):
            # Post-process predicted matrix: clamp negatives to zero.
            pred_matrix = fin[i]
            pred_matrix[pred_matrix < 0] = 0

            # Vectorize matrices
            pred_1d_list.append(MatrixVectorizer.vectorize(pred_matrix))
            gt_1d_list.append(MatrixVectorizer.vectorize(real[i]))

    # Concatenate flattened matrices
    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    overall_mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)


    if verbose:
        print("Overall MAE: ", overall_mae)
        print("Pearson Correlation Coefficient (PCC): ", pcc)
        print("Jensen-Shannon Distance: ", js_dis)


    return overall_mae, pcc, js_dis

def avg_shortest_path(graph):
    """
    Compute the average shortest path length on the largest connected component.
    """
    if nx.is_connected(graph):
        return nx.average_shortest_path_length(graph, weight="weight")
    else:
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        return nx.average_shortest_path_length(subgraph, weight="weight")

def process_single_sample(pred_matrix, gt_matrix):
    """
    Process one pair of predicted and ground truth matrices.
    Returns:
        vectorized_pred: 1D numpy array of vectorized predicted matrix
        vectorized_gt: 1D numpy array of vectorized ground truth matrix
        mae_bc: MAE for betweenness centrality
        mae_ec: MAE for eigenvector centrality
        mae_pc: MAE for PageRank centrality
        clustering_error: |avg_clustering(pred) - avg_clustering(gt)|
        small_worldness_error: |(C_pred/L_pred) - (C_gt/L_gt)|
    """
    # Copy to avoid modifying original and clamp negatives to zero
    pred_matrix = np.copy(pred_matrix)
    pred_matrix[pred_matrix < 0] = 0

    # Convert adjacency matrices to NetworkX graphs
    pred_graph = nx.from_numpy_array(pred_matrix, create_using=nx.Graph())
    gt_graph = nx.from_numpy_array(gt_matrix, create_using=nx.Graph())

    # Compute centrality measures
    pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
    pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
    pred_pc = nx.pagerank(pred_graph, weight="weight")

    gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
    gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
    gt_pc = nx.pagerank(gt_graph, weight="weight")

    # Convert centrality dictionaries to lists
    pred_bc_values = list(pred_bc.values())
    pred_ec_values = list(pred_ec.values())
    pred_pc_values = list(pred_pc.values())

    gt_bc_values = list(gt_bc.values())
    gt_ec_values = list(gt_ec.values())
    gt_pc_values = list(gt_pc.values())

    # Compute MAEs for centrality measures
    mae_bc = mean_absolute_error(pred_bc_values, gt_bc_values)
    mae_ec = mean_absolute_error(pred_ec_values, gt_ec_values)
    mae_pc = mean_absolute_error(pred_pc_values, gt_pc_values)

    # -----------------------------------------------------
    # 2) Unweighted Graphs: for clustering, small-worldness, etc.
    #    We treat any positive entry as an edge.
    # -----------------------------------------------------
    pred_unweighted = (pred_matrix > 0).astype(int)
    gt_unweighted   = (gt_matrix > 0).astype(int)

    G_pred_uw = nx.from_numpy_array(pred_unweighted, create_using=nx.Graph())
    G_gt_uw   = nx.from_numpy_array(gt_unweighted,   create_using=nx.Graph())

    # -- 2A) Clustering MAE --
    clust_pred = nx.clustering(G_pred_uw)
    clust_gt   = nx.clustering(G_gt_uw)
    clust_mae  = mean_absolute_error(list(clust_pred.values()),
                                    list(clust_gt.values()))

    # # -- 2B) Small-worldness difference (sigma) --
    # # It's possible for these calls to fail on disconnected or tiny graphs
    # try:
    #     sigma_pred = nx.algorithms.smallworld.sigma(G_pred_uw, niter=3, nrand=1)
    # except:
    #     sigma_pred = np.nan
    # try:
    #     sigma_gt = nx.algorithms.smallworld.sigma(G_gt_uw, niter=3, nrand=1)
    # except:
    #     sigma_gt = np.nan

    # if not np.isnan(sigma_pred) and not np.isnan(sigma_gt):
    #     sw_diff = abs(sigma_pred - sigma_gt)
    # else:
    #     sw_diff = np.nan
    # clust_mae = 0
    # sw_diff = 0

    # -- 2D) Degree MAE --
    deg_pred = [G_pred_uw.degree(n) for n in G_pred_uw.nodes()]
    deg_gt   = [G_gt_uw.degree(n)   for n in G_gt_uw.nodes()]
    deg_mae  = mean_absolute_error(deg_pred, deg_gt) / 100.0

    # Vectorize matrices
    vectorized_pred = MatrixVectorizer.vectorize(pred_matrix)
    vectorized_gt = MatrixVectorizer.vectorize(gt_matrix)
    return (vectorized_pred, vectorized_gt, mae_bc, mae_ec, mae_pc,
            clust_mae, deg_mae)

def compute_metrics_parallel(fin, real, verbose=True, n_jobs=8):
    """
    Compute evaluation metrics in parallel.
    Args:
        fin: List of predicted matrices
        real: List of ground-truth matrices
        verbose: Boolean to print metrics after computation
        n_jobs: Number of parallel processes (default: number of CPUs)
    Returns:
        overall_mae, pcc, js_dis, mae_pc, mae_ec, mae_bc
    """
    # Prepare arguments as a list of tuples for each sample
    args = list(zip(fin, real))

    with parallel_backend('loky'):
        # Use Joblib to process samples in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_sample)(pred, gt)
            for pred, gt in tqdm(args, desc="Processing samples", total=len(args))
        )

    # Unpack results
    pred_1d_list, gt_1d_list = [], []
    mae_bc_list, mae_ec_list, mae_pc_list = [], [], []
    clustering_errors = []
    small_worldness_errors = []
    degree_errors = []

    for res in results:
        vectorized_pred, vectorized_gt, mae_bc, mae_ec, mae_pc, clustering_error, degree_error = res
        pred_1d_list.append(vectorized_pred)
        gt_1d_list.append(vectorized_gt)
        mae_bc_list.append(mae_bc)
        mae_ec_list.append(mae_ec)
        mae_pc_list.append(mae_pc)
        clustering_errors.append(clustering_error)
        # small_worldness_errors.append(small_worldness_error)
        degree_errors.append(degree_error)

    # Concatenate flattened matrices
    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    overall_mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)

    avg_mae_bc = np.mean(mae_bc_list)
    avg_mae_ec = np.mean(mae_ec_list)
    avg_mae_pc = np.mean(mae_pc_list)
    avg_clustering_error = np.mean(clustering_errors)
    # avg_small_worldness_error = np.mean(small_worldness_errors)
    avg_degree_error = np.mean(degree_errors) 

    if verbose:
        print("Overall MAE:", overall_mae)
        print("Pearson Correlation Coefficient (PCC):", pcc)
        print("Jensen-Shannon Distance:", js_dis)
        print("Average MAE Betweenness Centrality:", avg_mae_bc)
        print("Average MAE Eigenvector Centrality:", avg_mae_ec)
        print("Average MAE PageRank Centrality:", avg_mae_pc)
        print("Average Clustering MAE:", avg_clustering_error)
        # print("Average Small-Worldness:", avg_small_worldness_error)
        print("Average Degree MAE:", avg_degree_error)

    return (overall_mae, pcc, js_dis, avg_mae_pc, avg_mae_ec, avg_mae_bc,
            avg_clustering_error, avg_degree_error)


def get_system_usage():
    """
    Return a dictionary with current system usage:
      - RAM_MB: Resident memory usage in MB.
      - CPU_percent: Overall CPU usage percentage.
      - VRAM_MB: GPU memory allocated in MB (None if no CUDA GPU available).
    """
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / (1024 ** 2)
    
    # CPU usage percentage over a 1-second interval
    cpu_usage = psutil.cpu_percent(interval=1)
    
    # Check for GPU availability and get VRAM usage if available
    vram_usage_mb = None
    if torch.cuda.is_available():
        vram_usage_mb = torch.cuda.memory_allocated() / (1024 ** 2)

        device = torch.device("cuda")
        free, total = torch.cuda.mem_get_info(device)
        vram_usage_mb = (total - free) / (1024 ** 2)

    return {"RAM_MB": ram_usage_mb, "CPU_percent": cpu_usage, "VRAM_MB": vram_usage_mb}


def plot_metrics(metrics_by_fold, k=1):
    """ 
    Plot the evaluation metrics for the model predictions.
    Supports k-fold metrics. If k > 1, it plots the average for all metrics
    for all folds.
    Args:
        metrics_by_fold: Dictionary of metrics for each fold:
                        MAE: Overall Mean Absolute Error
                        PCC: Pearson Correlation Coefficient
                        JSD: Jensen-Shannon Distance
                        mae_pc: Mean Absolute Error for PageRank Centrality
                        mae_ec: Mean Absolute Error for Eigenvector Centrality
                        mae_bc: Mean Absolute Error for Betweenness Centrality
                        title: Title of the plot (defaults to 'Evaluation Metrics')
        k: Number of folds (default: 1)
    """
    import math

    colors = ['mediumorchid', 'orchid', 'hotpink', 'magenta', 'deeppink', 'darkviolet', 'blueviolet', 'indigo']
    ncols = 2
    nrows = math.ceil((k+1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10*nrows+10, 8*nrows))

    ax_positions = [(i, j) for i in range(nrows) for j in range(ncols)]
    if ((k+1) % ncols) or (k==1):
        ax_positions.remove((nrows-1,1))
        if k>1: fig.delaxes(axes[nrows-1,1])
        else: fig.delaxes(axes[1])

    # plot metrics of each fold
    for i in range(k):
        pos = ax_positions[i]
        if k>1:
            metrics = metrics_by_fold[i]
            axes[pos].set_title(f'Fold {i+1}')
        if k<2:
            metrics = metrics_by_fold
            pos=0
            axes[pos].set_title('Evaluation Metrics')
        axes[pos].bar(0, metrics['MAE'], color=colors[0])
        axes[pos].bar(1, metrics['PCC'], color=colors[1])
        axes[pos].bar(2, metrics['JSD'], color=colors[2])
        axes[pos].bar(3, metrics['mae_pc'], color=colors[3])
        axes[pos].bar(4, metrics['mae_ec'], color=colors[4])
        axes[pos].bar(5, metrics['mae_bc'], color=colors[5])
        axes[pos].bar(6, metrics['mae_clst'], color=colors[6])
        axes[pos].bar(7, metrics['mae_deg'], color=colors[7])
        axes[pos].set_xticks([0, 1, 2, 3, 4, 5, 6, 7], ['MAE', 'PCC', 'JSD', 'MAE (PC)', 'MAE (EC)', 'MAE (BC)', 'MAE (CLS)', 'MAE (DEG)'])

    if k>1:
        # plot average metrics
        avg_metrics = { "MAE": np.mean([metrics_by_fold[i]['MAE'] for i in range(k)]),
                        "PCC": np.mean([metrics_by_fold[i]['PCC'] for i in range(k)]),
                        "JSD": np.mean([metrics_by_fold[i]['JSD'] for i in range(k)]),
                        "mae_pc": np.mean([metrics_by_fold[i]['mae_pc'] for i in range(k)]),
                        "mae_ec": np.mean([metrics_by_fold[i]['mae_ec'] for i in range(k)]),
                        "mae_bc": np.mean([metrics_by_fold[i]['mae_bc'] for i in range(k)]),
                        "mae_clst": np.mean([metrics_by_fold[i]['mae_clst'] for i in range(k)]),
                        "mae_deg": np.mean([metrics_by_fold[i]['mae_deg'] for i in range(k)])}
        
        metrics_std = { "MAE": np.std([metrics_by_fold[i]['MAE'] for i in range(k)]),
                        "PCC": np.std([metrics_by_fold[i]['PCC'] for i in range(k)]),
                        "JSD": np.std([metrics_by_fold[i]['JSD'] for i in range(k)]),
                        "mae_pc": np.std([metrics_by_fold[i]['mae_pc'] for i in range(k)]),
                        "mae_ec": np.std([metrics_by_fold[i]['mae_ec'] for i in range(k)]),
                        "mae_bc": np.std([metrics_by_fold[i]['mae_bc'] for i in range(k)]),
                        "mae_clst": np.std([metrics_by_fold[i]['mae_clst'] for i in range(k)]),
                        "mae_deg": np.std([metrics_by_fold[i]['mae_deg'] for i in range(k)])}
        
        pos = ax_positions[-1]
        axes[pos].set_title('Average Evaluation Metrics')
        axes[pos].bar(0, avg_metrics['MAE'], color=colors[0])
        axes[pos].bar(1, avg_metrics['PCC'], color=colors[1])
        axes[pos].bar(2, avg_metrics['JSD'], color=colors[2])
        axes[pos].bar(3, avg_metrics['mae_pc'], color=colors[3])
        axes[pos].bar(4, avg_metrics['mae_ec'], color=colors[4])
        axes[pos].bar(5, avg_metrics['mae_bc'], color=colors[5])        
        axes[pos].bar(6, avg_metrics['mae_clst'], color=colors[6])
        axes[pos].bar(7, avg_metrics['mae_deg'], color=colors[7])
        axes[pos].errorbar(0, avg_metrics['MAE'], yerr=metrics_std['MAE'], color='black')
        axes[pos].errorbar(1, avg_metrics['PCC'], yerr=metrics_std['PCC'], color='black')
        axes[pos].errorbar(2, avg_metrics['JSD'], yerr=metrics_std['JSD'], color='black')
        axes[pos].errorbar(3, avg_metrics['mae_pc'], yerr=metrics_std['mae_pc'], color='black')
        axes[pos].errorbar(4, avg_metrics['mae_ec'], yerr=metrics_std['mae_ec'], color='black')
        axes[pos].errorbar(5, avg_metrics['mae_bc'], yerr=metrics_std['mae_bc'], color='black')
        axes[pos].errorbar(6, avg_metrics['mae_clst'], yerr=metrics_std['mae_clst'], color='black')
        axes[pos].errorbar(7, avg_metrics['mae_deg'], yerr=metrics_std['mae_deg'], color='black')
        axes[pos].set_xticks([0, 1, 2, 3, 4, 5, 6, 7], ['MAE', 'PCC', 'JSD', 'MAE (PC)', 'MAE (EC)', 'MAE (BC)', 'MAE (CLS)', 'MAE (DEG)'])
    
    plt.show()

############################## MODEL FUNCTIONS #############################

def run_epoch(model, data_loader, optim, device, train=True):
    """
    Run one epoch of training or evaluation.
    Args:
        model: Model to train or evaluate
        data_loader: DataLoader object
        optim: Optimizer object
        device: Device to run the model on
        train: Boolean to specify training or evaluation
    Returns:
        epoch_loss: Loss for the current epoch
    """
    model.train() if train else model.eval()
    loss_ema = None
    ema_decay = 0.99
    batch_count, epoch_loss = 0, 0.0

    for c, x in data_loader:
        batch_count += 1
        if train:
            optim.zero_grad()
        c = c.squeeze(1).to(device)
        x = x.to(device)
        # print(x.shape, c.shape)
        loss = model(x, c)
        if train:
            loss.backward()
            loss_ema = loss.item() if loss_ema is None else ((loss_ema * ema_decay) + loss.item() * (1 - ema_decay))
            optim.step()
        epoch_loss += loss.item()

    epoch_loss /= batch_count
    return epoch_loss

def run_epoch_MLP(model, data_loader, optim, device, train=True):
    """
    Run one epoch of training or evaluation.
    Args:
        model: Model to train or evaluate
        data_loader: DataLoader object
        optim: Optimizer object
        device: Device to run the model on
        train: Boolean to specify training or evaluation
    Returns:
        epoch_loss: Loss for the current epoch
    """
    model.train() if train else model.eval()
    loss_ema = None
    ema_decay = 0.99
    batch_count, epoch_loss = 0, 0.0
    criterion = nn.MSELoss()
    for lr, hr in data_loader:
        batch_count += 1
        if train:
            optim.zero_grad()
        hr = hr.squeeze(1).to(device)
        lr = lr.to(device)
        # print(x.shape, c.shape)
        pred_hr = model(lr)
        loss = criterion(pred_hr, hr)

        if train:
            loss.backward()
            loss_ema = loss.item() if loss_ema is None else ((loss_ema * ema_decay) + loss.item() * (1 - ema_decay))
            optim.step()
        epoch_loss += loss.item()

    epoch_loss /= batch_count
    return epoch_loss


def train_model(model, n_epoch, train_loader, test_loader, optim, lr, device, patience = 50, scheduler=None, save=None, baseline=False):
    """
    Train and validate the model for a specified number of epochs.
    Args:
        model: Model
        n_epoch: Number of epochs
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation data
        optim: Optimizer
        lr: Learning rate
        device: Device to run the model on
        patience: Patience for early stopping (default: 50)
        save: Boolean to save the best model state or not (default: False)
        baseline: Boolean to train baseline MLP model
    Returns:
        model: Trained model
        train_loss_history: List of training losses for all epochs
        val_loss_history: List of validation losses for all epochs
    """
    best_val_loss = float('inf')
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    system_history = []

    pbar = tqdm(range(n_epoch))
    for ep in pbar:
        # Train for one epoch
        if scheduler is None:
            optim.param_groups[0]['lr'] = lr * (1 - ep / n_epoch)
        if baseline:
            train_loss_epoch = run_epoch_MLP(model, train_loader, optim, device, train=True)
        else:
            train_loss_epoch = run_epoch(model, train_loader, optim, device, train=True)
        train_loss_history.append(train_loss_epoch)

        # Evaluate on validation set
        with torch.no_grad():
            if baseline:
                val_loss_epoch = run_epoch_MLP(model, test_loader, optim, device, train=False)
            else:
                val_loss_epoch = run_epoch(model, test_loader, optim, device, train=False)
        val_loss_history.append(val_loss_epoch)

        if scheduler is not None:
            scheduler.step(val_loss_epoch)

        # System usage in training loop:
        system_usage = get_system_usage()
        ram_use = system_usage["RAM_MB"]
        cpu_use = system_usage["CPU_percent"]
        vram_use = system_usage["VRAM_MB"]
        system_history.append(system_usage)
        pbar.set_description(f"Epoch {ep} - train_loss: {train_loss_epoch:.4f}, val_loss: {val_loss_epoch:.4f}, RAM: {ram_use:.2f} MB, CPU: {cpu_use} %, VRAM: {vram_use:.2f} MB")
        pbar.write(f"Epoch {ep}, train_loss: {train_loss_epoch:.4f}, val_loss: {val_loss_epoch:.4f}")

        # Early Stopping Check
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            # Optionally save the best model state
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                pbar.write(f"Early stopping triggered after {ep} epochs.")
                break
    if save is not None:
        torch.save(best_model_state, save)
    return model, train_loss_history, val_loss_history, system_history


def eval_model(model, test_loader, device):
    """"
    Evaluate the model on the test set.
    Args:
        model: Model
        test_loader: DataLoader for test data
        device: Device to run the model on
    Returns:
        model: Model after evaluation
        fin: List of generated matrices
        real: List of ground-truth matrices
    """
    model.eval()
    fin = []
    real = []
    system_history = []

    with torch.no_grad():
        for lr, hr in test_loader: 
            # print(lr.shape, hr.shape)
            real.extend(hr.cpu().squeeze(1).numpy()) 

            # Move LR input to device
            lr = lr.to(device)

            # Generate samples
            x_gen = model.sample(
                n_sample=lr.size(0),
                size=(1, 268, 268),
                device=device,
                context_i_1=lr,  # Use LR as conditioning
                guide_w=0.5      # Classifier-free guidance scale
            ).clamp(0,1)
            # print(x_gen.shape)
            fin.extend(x_gen.cpu().squeeze(1).numpy())
            system_usage = get_system_usage()
            system_history.append(system_usage)

    return model, fin, real, system_history

def eval_MLP(model, test_loader, device):
    """"
    Evaluate MLP on the test set.
    Args:
        model: Model
        test_loader: DataLoader for test data
        device: Device to run the MLP on
    Returns:
        model: MLP after evaluation
        fin: List of generated matrices
        real: List of ground-truth matrices
    """
    model.eval()
    fin = []
    real = []
    system_history = []

    with torch.no_grad():
        for lr, hr in test_loader:  # Assuming test_loader yields (lr, hr) pairs
            # print(lr.shape, hr.shape)
            # Store ground truth
            real.extend(hr.cpu().squeeze(1).numpy())  # Assuming HR is the target

            # Move LR input to device
            lr = lr.to(device)

            # Generate samples
            x_gen = model(lr).clamp(0,1)
            # print(x_gen.shape)
            fin.extend(x_gen.cpu().numpy())
            system_usage = get_system_usage()
            system_history.append(system_usage)
    fin = np.array(fin)
    fin = fin.reshape(fin.shape[0], 268, 268)
    real = np.array(real)
    real = real.reshape(fin.shape[0], 268, 268)
    return model, fin, real, system_history


def run_kfold(DDPM, UnetModel, n_feat, patience, n_epoch, lr, device, dataset, test_lr_tensors, k=3, seed=42, plot=True, save=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Perform fold split
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    metrics_by_fold = {i:{ "MAE": None,
                            "PCC": None,
                            "JSD": None,
                            "mae_pc": None,
                            "mae_ec": None,
                            "mae_bc": None,
                            "mae_clst": None, 
                            "mae_deg": None} for i in range(k)}
    save_name = None

    run_history = {}
    predictions = []
    # Run 3-fold
    for i, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1):
        print(f"Fold {i}:")
        start_fold = time.time()

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        eps_model = UnetModel(in_channels=1,
                        n_feat=n_feat).to(device)
        
        print("Denoiser model total trainable parameters:", sum(p.numel() for p in eps_model.parameters()))

        # Initialise model
        model = DDPM(denoising_model=eps_model,
                    beta1=0.0001,
                    beta2=0.02,
                    n_T=100,
                    drop_prob=0.2,
                    lr=lr,
                    device=device).to(device)

        if save is not None:
            folder_path = f"results/{save}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_name = f"{folder_path}/model_fold_{i}.pt"

        # Train model 
        model, train_loss_history, val_loss_history, system_history = train_model(model,
                                                                n_epoch,
                                                                train_loader,
                                                                val_loader,
                                                                torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2),
                                                                lr,
                                                                device,
                                                                patience = patience,
                                                                save=save_name)
        

        model, fin, real, system_history_eval = eval_model(model, val_loader, device)
        # fast_metrics(fin, real, verbose=True)
        # return
        overall_mae, pcc, js_dis, mae_pc, mae_ec, mae_bc, mae_clst, mae_deg = compute_metrics_parallel(fin, real, verbose=True)
        time_taken = time.time() - start_fold

        metrics_by_fold[i-1] = {"MAE": overall_mae,
                                "PCC": pcc,
                                "JSD": js_dis,
                                "mae_pc": mae_pc,
                                "mae_ec": mae_ec,
                                "mae_bc": mae_bc,
                                "mae_clst": mae_clst,
                                "mae_deg": mae_deg,}
        
        run_history[i-1] = {"train_loss": train_loss_history,
                          "val_loss": val_loss_history,
                          "system_usage": system_history,
                          "system_usage_eval": system_history_eval,
                          "time": time_taken}
        
        if save:
            pred1d_list = []
            for adj_mat in fin:
                pred1d_list.append(MatrixVectorizer.vectorize(adj_mat))

            # Concatenate all predictions
            predictions = np.concatenate(pred1d_list)

            file_name = f"{folder_path}/predictions_fold_{i}.csv"
            create_kaggle_submission_file(predictions, filename=file_name)
        if plot:
            plot_metrics(metrics_by_fold[i-1], k=1)
    if save:
        results = {
            "metrics_by_fold": metrics_by_fold,
            "run_history": run_history
        }

        json_file_name = f"{folder_path}/metrics.csv"
        # Write the dictionary to the JSON file
        with open(json_file_name, "w") as f:
            json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        print(f"Results saved to {json_file_name}")

    return metrics_by_fold, run_history


def run_kfold_MLP(MLP, args, patience, n_epoch, lr, device, dataset, test_lr_tensors, k=3, seed=42, plot=True, save=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Perform fold split
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    metrics_by_fold = {i:{ "MAE": None,
                            "PCC": None,
                            "JSD": None,
                            "mae_pc": None,
                            "mae_ec": None,
                            "mae_bc": None,
                            "mae_clst": None, 
                            "mae_deg": None} for i in range(k)}
    save_name = None

    run_history = {}
    predictions = []
    # Run 3-fold
    for i, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1):
        print(f"Fold {i}:")
        start_fold = time.time()

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # Initialise model
        model = MLP(args).to(device)

        print("MLP model total trainable parameters:", sum(p.numel() for p in model.parameters()))


        if save is not None:
            folder_path = f"results/{save}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_name = f"{folder_path}/model_fold_{i}.pt"

        # Train model 
        model, train_loss_history, val_loss_history, system_history = train_model(model,
                                                                n_epoch,
                                                                train_loader,
                                                                val_loader,
                                                                torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2),
                                                                lr,
                                                                device,
                                                                patience=patience,
                                                                save=save_name,
                                                                baseline=True)
        

        model, fin, real, system_history_eval = eval_MLP(model, val_loader, device)

        overall_mae, pcc, js_dis, mae_pc, mae_ec, mae_bc, mae_clst, mae_deg = compute_metrics_parallel(fin, real, verbose=True)
        time_taken = time.time() - start_fold

        metrics_by_fold[i-1] = {"MAE": overall_mae,
                                "PCC": pcc,
                                "JSD": js_dis,
                                "mae_pc": mae_pc,
                                "mae_ec": mae_ec,
                                "mae_bc": mae_bc,
                                "mae_clst": mae_clst,
                                "mae_deg": mae_deg,}
        
        run_history[i-1] = {"train_loss": train_loss_history,
                          "val_loss": val_loss_history,
                          "system_usage": system_history,
                          "system_usage_eval": system_history_eval,
                          "time": time_taken}
        
        if save:
            pred1d_list = []
            for adj_mat in fin:
                pred1d_list.append(MatrixVectorizer.vectorize(adj_mat))
                
            # Concatenate all predictions
            predictions = np.concatenate(pred1d_list)
            file_name = f"{folder_path}/predictions_fold_{i}.csv"
            create_kaggle_submission_file(predictions, filename=file_name)
        if plot:
            plot_metrics(metrics_by_fold[i-1], k=1)
    if save:
        results = {
            "metrics_by_fold": metrics_by_fold,
            "run_history": run_history
        }

        json_file_name = f"{folder_path}/metrics.csv"
        # Write the dictionary to the JSON file
        with open(json_file_name, "w") as f:
            json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        print(f"Results saved to {json_file_name}")

    return metrics_by_fold, run_history