import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset

from MatrixVectorizer import MatrixVectorizer
from models.DDPM_Schedule import DDPM, DDPMTopology
from models.unet import ContextUnet, ContextUnetGraph
from utils import *
from tqdm import tqdm 

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

def main():
    # Set a fixed random seed for reproducibility across multiple libraries
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Check for CUDA (GPU support) and set device accordingly
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups
        # Additional settings for ensuring reproducibility on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # !nvidia-smi
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available.")
    else:
        device = torch.device("cpu")
        print("CUDA not available.")
    
    print("Using", device)


    lr_data_path_train = 'data/lr_train.csv'
    lr_data_path_test = 'data/lr_test.csv'
    hr_data_path = 'data/hr_train.csv'
    
    # load data
    df_lr_train = pd.read_csv(lr_data_path_train)
    df_lr_test = pd.read_csv(lr_data_path_test)
    df_hr_train = pd.read_csv(hr_data_path)
    
    lr_matrices = vectorize_matrix(df_lr_train, 160)
    hr_matrices = vectorize_matrix(df_hr_train, 268)


    patch_size_lr = 20
    hr_all = np.mean(hr_matrices, axis=0, keepdims=True).squeeze()
    lr_matrices_matched = []
    for lr_mat in lr_matrices:
        lr_matrices_matched.append(patch_histogram_matching(lr_mat, hr_all, patch_size_lr, nbins=256))
    lr_matrices = np.array(lr_matrices_matched)

    hr_matrices.extend(hr_matrices)
    # convert data to tensors
    lr_tensors = torch.stack([torch.FloatTensor(x) for x in lr_matrices]).to(device)
    hr_tensors = torch.stack([torch.FloatTensor(x) for x in hr_matrices]).to(device)
    
    # needed dimension to use ddpm
    hr_tensors = hr_tensors.unsqueeze(1)
    
    dataset = TensorDataset(lr_tensors, hr_tensors)
    
    # split training/test sets (80% train, 20% test)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    lr = 1e-4
    # n_feat = 80 # 64
    n_feat = 32 # 64
    
    eps_model = ContextUnet(in_channels=1,
                    n_feat=n_feat,
                    n_classes=160**2).to(device)
        
    # model = DDPM(denoising_model=eps_model,
    #             beta1=0.0001,
    #             beta2=0.02,
    #             n_T=100,
    #             drop_prob=0.2,
    #             lr=lr,
    #             device=device).to(device)

    model = DDPMTopology(denoising_model=eps_model,
                    beta1=0.0001,
                    beta2=0.02,
                    n_T=100,
                    drop_prob=0.2,
                    lr=lr,
                    device=device).to(device)

    patience = 5
    n_epoch = 200

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3)

    print(model(
        torch.randn(2,1,268,268).to(device).clamp(0),
        torch.randn(2,160,160).to(device).clamp(0),
    ))

    print("Denoiser model total trainable parameters:", sum(p.numel() for p in eps_model.parameters()))


    model, train_loss_history, val_loss_history = train_model(model,
                                                            n_epoch,
                                                            train_loader,
                                                            val_loader,
                                                            optim,
                                                            scheduler,
                                                            lr,
                                                            device,
                                                            patience = patience,
                                                            save=None)
    
    # model, fin, real = eval_model(model, val_loader, device)
    
    # overall_mae, pcc, js_dis = fast_metrics(fin, real, verbose=True)

    # Load test data
    test_lr_matrices = vectorize_matrix(df_lr_test, 160)
    
    hr_all = np.mean(hr_matrices, axis=0, keepdims=True).squeeze()
    # print(hr_all.shape)
    patch_size_lr = 40
    lr_matrices_matched = []
    for lr_mat in test_lr_matrices:
        lr_matrices_matched.append(patch_histogram_matching(lr_mat, hr_all, patch_size_lr, nbins=256))
    test_lr_matrices = np.array(lr_matrices_matched)

    test_lr_tensors = torch.stack([torch.FloatTensor(x) for x in test_lr_matrices]).to(device)

    # Generate predictions
    model.eval()
    pred_list = []

    with torch.no_grad():
        for sample in tqdm(test_lr_tensors):
            # Use the sample method to generate HR graphs
            sample = sample.unsqueeze(0)

            # Generate samples
            generated_hr = model.sample(
                n_sample=sample.size(0),
                size=(1, 268, 268),
                device=device,
                context_i_1=sample,  # Use LR as conditioning
                guide_w=0.5,      # Classifier-free guidance scale
                use_cbar=False,
            ).clamp(0, 1)
            pred_list.append(generated_hr.squeeze().cpu().numpy())

    print(generated_hr.shape)

    print(f"Generated {len(pred_list)} predictions")

    # Convert predictions to 1D vectors for submission
    pred1d_list = []
    for adj_mat in pred_list:
        pred1d_list.append(MatrixVectorizer.vectorize(adj_mat))

    # Concatenate all predictions
    pred_1d = np.concatenate(pred1d_list)
    print(f"Final prediction shape: {pred_1d.shape}")

    # Create submission file
    create_kaggle_submission_file(pred_1d, filename='submission.csv')
    print("Created submission file: submission.csv")

if __name__ == "__main__":
    main()