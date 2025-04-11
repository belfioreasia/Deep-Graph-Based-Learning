import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import networkx as nx
import numpy as np

def ddpm_schedules(beta1, beta2, T):

    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """

    assert beta1 < beta2 < 1.0
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,  
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

def differentiable_pearson(x, y, eps=1e-8):
    # Ensure the tensors are float and have the same shape
    x = x.float()
    y = y.float()
    
    # Compute means
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    
    # Center the variables
    xm = x - mean_x
    ym = y - mean_y
    
    # Compute covariance and standard deviations
    covariance = torch.mean(xm * ym)
    std_x = torch.sqrt(torch.mean(xm ** 2) + eps)
    std_y = torch.sqrt(torch.mean(ym ** 2) + eps)
    
    # Compute the Pearson correlation coefficient
    return covariance / (std_x * std_y)

############################## LOSS FUNCTIONS #############################

def degree_loss(x0_est, x0):
    """
    x0_est, x0: shape (batch_size, 1, n_t, n_t) or something similar
    Returns: A (batch_size,) or scalar with the topological discrepancy
    """

    # Convert each adjacency to a node-degree vector by summing across the row dimension:
    degrees_est = x0_est.sum(dim=-1).squeeze(1)  # sum over width
    degrees_true = x0.sum(dim=-1).squeeze(1)

    # Check for NaNs or empty tensors
    if torch.isnan(degrees_est).any() or torch.isnan(degrees_true).any():
        print("Warning: NaN values found in degree vectors. Replacing with 0.")
        degrees_est = torch.nan_to_num(degrees_est)
        degrees_true = torch.nan_to_num(degrees_true)

    if degrees_est.nelement() == 0 or degrees_true.nelement() == 0:
        print("Warning: One of the degree tensors is empty!")
        degrees_est = torch.zeros_like(degrees_true)  # Fill with zeros

    # Then compute L1 or MSE difference in degrees:
    deg_loss = F.mse_loss(degrees_est, degrees_true, reduction="mean")
    #print('Degree loss', deg_loss.shape)
    # Possibly do the same for columns if adjacency is not symmetric, or sum them if it is symmetrical. 
    # Then return the result.

    return deg_loss

def eigenvector_centrality_loss(x0_est, x0):
    """
    Compute the eigenvector centrality loss between estimated and true adjacency matrices.
    
    Args:
        x0_est (torch.Tensor): Estimated adjacency matrices (batch_size, 1, n_t, n_t).
        x0 (torch.Tensor): Ground truth adjacency matrices (batch_size, 1, n_t, n_t).
    
    Returns:
        torch.Tensor: Loss values for each batch element (batch_size,).
    """
    # x0_est = x0_est.unsqueeze(0)  # Add singleton dimension for batch size
    # x0 = x0.unsqueeze(0)  # Add singleton dimension for batch size
    x0_est = x0_est.squeeze(1)
    x0 = x0.squeeze(1)
    batch_size = x0_est.shape[0]
    # print(x0_est.shape, x0.shape)
    ec_losses = []

    for i in range(batch_size):
        # Convert adjacency matrices to NumPy arrays (assuming tensors)
        adj_est = x0_est[i].cpu().detach().numpy()  # Convert from tensor to numpy
        adj_true = x0[i].cpu().detach().numpy()

        # Convert adjacency matrices to graphs
        G_est = nx.from_numpy_array(adj_est, create_using=nx.Graph())
        G_true = nx.from_numpy_array(adj_true, create_using=nx.Graph())

        try:
            # Compute eigenvector centrality with a higher iteration limit
            ec_est_dict = nx.eigenvector_centrality(G_est, weight="weight", max_iter=100, tol=1e-6)
            ec_true_dict = nx.eigenvector_centrality(G_true, weight="weight", max_iter=100, tol=1e-6)
        except nx.NetworkXError:
            print(f"Eigenvector centrality computation failed at batch index {i}")
            ec_losses.append(torch.tensor(float('inf')))  # Assign high loss if failure occurs
            continue

        # Convert dict values to sorted numpy arrays
        ec_est_values = np.array(list(ec_est_dict.values()))
        ec_true_values = np.array(list(ec_true_dict.values()))

        # Convert to tensors for loss computation
        ec_est_tensor = torch.tensor(ec_est_values, dtype=torch.float32, device='cpu')
        ec_true_tensor = torch.tensor(ec_true_values, dtype=torch.float32, device='cpu')

        # Compute Mean Squared Error (MSE) loss
        loss = F.mse_loss(ec_est_tensor, ec_true_tensor, reduction="mean")
        ec_losses.append(loss)
    #print('EC Loss', torch.stack(ec_losses).shape)
    return torch.stack(ec_losses)  # Return tensor of losses for each batch

class DDPMTopology(nn.Module):
    def __init__(self,
                 denoising_model,
                 beta1,
                 beta2,
                 n_T,
                 drop_prob=0.1,
                 lr=1e-4,
                 device="cpu"):
        super(DDPMTopology, self).__init__()
        self.nn_model = denoising_model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        for k, v in ddpm_schedules(self.beta1, self.beta2, n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.device = device
        
        # Weight for topology loss
        self.lambda_topo = 0.1
        self.loss_mae_recon = nn.L1Loss()
    
    def forward(self, x, c):

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )
        context_mask = torch.bernoulli(torch.zeros_like(_ts) + self.drop_prob).to(self.device)

        # Predict the noise using the denoising model.
        pred_noise = self.nn_model(x_t, c, _ts / self.n_T, context_mask) 

        # print(pred_noise.mean(), noise.mean())
        # Standard diffusion loss: compare predicted noise to actual noise.
        noise_loss = self.loss_mse(noise, pred_noise)
        
        # Reconstruct the predicted clean image using the diffusion reconstruction formula.
        x_hat = (x_t - self.sqrtmab[_ts, None, None, None] * pred_noise) / self.sqrtab[_ts, None, None, None]
        
        # Compute topology-aware loss  
        mae = self.loss_mae_recon(x_hat, x)
        corr = 1 - differentiable_pearson(x_hat, x)
        # eigen_loss = eigenvector_centrality_loss(x_hat.clamp(0, 1), x).mean()

        # Total loss: original noise loss plus topology loss (weighted).
        # total_loss = noise_loss*0.25 + mae * 0.25  + corr * 0.25 + eigen_loss * 0.25
        total_loss = noise_loss*0.7 + mae * 0.1  + corr * 0.2
        # total_loss = noise_loss
        
        return total_loss
        

    def sample(self, n_sample, size, device, context_i_1, guide_w=0.0, use_cbar=True):

        x_i = torch.randn(n_sample, *size).to(device)
        c_i = context_i_1.view(x_i.shape[0], -1)
        context_mask = torch.zeros((2 * c_i.shape[0])).to(device)
        c_i = c_i.repeat(2, 1)

        context_mask[n_sample:] = 1.  # makes second half of batch context free
        # print()
        if use_cbar:
            for i in tqdm(range(self.n_T, 0, -1)):
                t_is = torch.tensor([i / self.n_T]).to(device)
                t_is = t_is.repeat(n_sample, 1, 1, 1)

                x_i = x_i.repeat(2, 1, 1, 1)
                t_is = t_is.repeat(2, 1, 1, 1)
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + guide_w) * eps1 - guide_w * eps2
                x_i = x_i[:n_sample]
                x_i = (
                        self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                        + self.sqrt_beta_t[i] * z
                )
        else:
            for i in range(self.n_T, 0, -1):
                t_is = torch.tensor([i / self.n_T]).to(device)
                t_is = t_is.repeat(n_sample, 1, 1, 1)

                x_i = x_i.repeat(2, 1, 1, 1)
                t_is = t_is.repeat(2, 1, 1, 1)
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + guide_w) * eps1 - guide_w * eps2
                x_i = x_i[:n_sample]
                x_i = (
                        self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                        + self.sqrt_beta_t[i] * z
                )
        return x_i


class DDPM(nn.Module):
    def __init__(self,
                 denoising_model,
                 beta1,
                 beta2,
                 n_T,
                 drop_prob=0.1,
                 lr=1e-4,
                 device="cpu"):
        super(DDPM, self).__init__()
        self.nn_model = denoising_model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        for k, v in ddpm_schedules(self.beta1, self.beta2, n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.device = device

    
    def forward(self, x, c):

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )
        context_mask = torch.bernoulli(torch.zeros_like(_ts) + self.drop_prob).to(self.device)

        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))
        # return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T))

    def sample(self, n_sample, size, device, context_i_1, guide_w=0.0, use_cbar=True):

        x_i = torch.randn(n_sample, *size).to(device)
        c_i = context_i_1.view(x_i.shape[0], -1)
        context_mask = torch.zeros((2 * c_i.shape[0])).to(device)
        c_i = c_i.repeat(2, 1)

        context_mask[n_sample:] = 1.  # makes second half of batch context free
        if use_cbar:
            for i in tqdm(range(self.n_T, 0, -1)):
                t_is = torch.tensor([i / self.n_T]).to(device)
                t_is = t_is.repeat(n_sample, 1, 1, 1)

                x_i = x_i.repeat(2, 1, 1, 1)
                t_is = t_is.repeat(2, 1, 1, 1)
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
                # eps = self.nn_model(x_i, c_i, t_is)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + guide_w) * eps1 - guide_w * eps2
                x_i = x_i[:n_sample]
                x_i = (
                        self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                        + self.sqrt_beta_t[i] * z
                )
        else:
            for i in range(self.n_T, 0, -1):
                t_is = torch.tensor([i / self.n_T]).to(device)
                t_is = t_is.repeat(n_sample, 1, 1, 1)

                x_i = x_i.repeat(2, 1, 1, 1)
                t_is = t_is.repeat(2, 1, 1, 1)
                z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
                eps = self.nn_model(x_i, c_i, t_is, context_mask)
                # eps = self.nn_model(x_i, c_i, t_is)
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + guide_w) * eps1 - guide_w * eps2
                x_i = x_i[:n_sample]
                x_i = (
                        self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                        + self.sqrt_beta_t[i] * z
                )
        return x_i

