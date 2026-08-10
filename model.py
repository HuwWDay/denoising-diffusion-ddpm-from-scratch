"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    return torch.linspace(beta_start, beta_end, T)

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    # TODO: return 1 - betas
    return 1-betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    # TODO: cumulative product of alphas
    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    return a.gather(0, t.long()).reshape(-1, 1, 1, 1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    alphabar = extract_into_batch(alphas_cumprod, t, x0)
    return torch.sqrt(alphabar)*x0 + torch.sqrt(1-alphabar)*noise

# Step 6 - build_diffusion_schedule
import torch
import torch.nn.functional as F

def build_diffusion_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    # TODO: build betas, alphas, alphas_cumprod and useful sqrts
    out = {}
    betas = linear_beta_schedule(T, beta_start, beta_end)
    out["betas"] = betas 
    alphas = alphas_from_betas(betas)
    out["alphas"] = alphas
    prodalpha = cumprod_alphas(alphas)
    out["alphas_cumprod"] = prodalpha
    out["sqrt_alphas_cumprod"] = torch.sqrt(prodalpha)
    out["sqrt_one_minus_alphas_cumprod"] = torch.sqrt(1-prodalpha)
    out["T"] = T 
    return out

# Step 7 - noise_prediction_loss
import torch
import torch.nn.functional as F

def noise_prediction_loss(noise_pred, noise):
    # TODO: MSE between predicted and true noise
    return ((noise-noise_pred)**2).mean()

# Step 8 - diffusion_training_loss
import torch
import torch.nn.functional as F

def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # TODO: q_sample -> model -> MSE(noise_pred, noise)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    noise_pred = model(x_t, t)
    return noise_prediction_loss(noise_pred, noise)

# Step 9 - timestep_embedding
import math
import torch
import torch.nn.functional as F

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        t: A 1D Tensor of shape (B,) containing timestep indices.
        dim: The dimension of the output embedding.
        max_period: Controls the minimum frequency of the embeddings.

    Returns:
        A Tensor of shape (B, dim) containing the positional embeddings.
    """
    half_dim = dim // 2
    
    # Calculate angular frequencies: exp(-log(max_period) * i / half_dim)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
    )
    
    # Outer product between timesteps and frequencies: (B, 1) * (1, half_dim) -> (B, half_dim)
    args = t[:, None].float() * freqs[None, :]
    
    # Concatenate sine and cosine components along the feature dimension
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    # Pad with a zero if the requested dimension is odd
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
        
    return embedding

# Step 10 - init_tiny_unet
import torch
import torch.nn.functional as F

def init_tiny_unet(in_ch: int = 1, hidden: int = 16, time_dim: int = 16, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    std = 0.02

    params = {
        "conv_in_w": torch.randn(hidden, in_ch, 3, 3) * std,
        "conv_in_b": torch.zeros(hidden),
        "time_mlp_w": torch.randn(hidden, time_dim) * std,
        "time_mlp_b": torch.zeros(hidden),
        "conv_mid_w": torch.randn(hidden, hidden, 3, 3) * std,
        "conv_mid_b": torch.zeros(hidden),
        "conv_out_w": torch.randn(in_ch, hidden, 3, 3) * std,
        "conv_out_b": torch.zeros(in_ch),
    }

    # Enable gradients for training
    for tensor in params.values():
        tensor.requires_grad_(True)

    return params

# Step 11 - tiny_unet_forward
import torch
import torch.nn.functional as F

def tiny_unet_forward(x: torch.Tensor, t: torch.Tensor, params: dict) -> torch.Tensor:
    # 1. Initial 3x3 convolution with padding=1 to preserve spatial dimensions
    h = F.conv2d(x, params['conv_in_w'], params['conv_in_b'], padding=1)
    
    # 2. Get sinusoidal timestep embeddings and project via Linear + ReLU
    time_dim = params['time_mlp_w'].shape[1]
    temb = timestep_embedding(t, time_dim)
    temb = F.relu(F.linear(temb, params['time_mlp_w'], params['time_mlp_b']))
    
    # 3. Add broadcasted time embedding to feature map: (B, C) -> (B, C, 1, 1)
    h = h + temb[:, :, None, None]
    
    # 4. Non-linearity followed by middle 3x3 convolution with padding=1
    h = F.relu(h)
    h = F.relu(F.conv2d(h, params['conv_mid_w'], params['conv_mid_b'], padding=1))
    
    # 5. Output 3x3 convolution to project back to image channels (in_ch)
    return F.conv2d(h, params['conv_out_w'], params['conv_out_b'], padding=1)

# Step 12 - make_blob_dataset
import torch

def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    
    radius = size // 4
    min_c = radius
    max_c = size - radius
    
    # Grid of coordinates (y, x) for distance calculations
    y_coords = torch.arange(size, dtype=torch.float32).view(1, size, 1)
    x_coords = torch.arange(size, dtype=torch.float32).view(1, 1, size)
    
    images = torch.zeros((n, 1, size, size), dtype=torch.float32)
    
    for i in range(n):
        # Draw center (cy, cx) within valid bounds
        center = torch.randint(min_c, max_c, (2,))
        cy, cx = center[0].item(), center[1].item()
        
        # Calculate squared distance from center: (y - cy)^2 + (x - cx)^2
        dist_sq = (y_coords - cy) ** 2 + (x_coords - cx) ** 2
        
        # Create filled disk mask (inclusive radius check)
        disk_mask = dist_sq <= (radius ** 2)
        images[i, 0] = disk_mask.float()
        
    return images

# Step 13 - ddpm_train_step
import torch
import torch.nn.functional as F

def ddpm_train_step(params: dict, x0: torch.Tensor, schedule: dict, lr: float = 1e-2, seed: int = 0) -> tuple[dict, float]:
    # 1. Seed RNG for reproducible t and noise sampling
    torch.manual_seed(seed)
    
    B = x0.shape[0]
    T = schedule['T']
    alphas_cumprod = schedule['alphas_cumprod']
    
    # 2. Sample timesteps t ~ Uniform{0..T-1} of shape (B,) on x0's device
    t = torch.randint(0, T, (B,), device=x0.device)
    
    # 3. Sample noise ~ N(0, I) matching shape and device of x0
    noise = torch.randn_like(x0)
    
    # 4. Zero existing gradients before forward/backward pass
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()
            
    # 5. Compute noise prediction loss using the model closure
    model_fn = lambda x, step_t: tiny_unet_forward(x, step_t, params)
    loss = diffusion_training_loss(model_fn, x0, t, noise, alphas_cumprod)
    
    # 6. Backpropagate gradients
    loss.backward()
    
    # 7. Functional SGD parameter update creating detached new leaf tensors
    new_params = {}
    for name, p in params.items():
        if p.grad is not None:
            p_new = (p - lr * p.grad).detach().requires_grad_(True)
        else:
            p_new = p.clone().detach().requires_grad_(True)
        new_params[name] = p_new
        
    return new_params, float(loss.item())

# Step 14 - train_ddpm
import torch
import torch.nn.functional as F

def train_ddpm(dataset: torch.Tensor, params: dict, schedule: dict, num_steps: int = 50, batch_size: int = 16, lr: float = 1e-2, seed: int = 0) -> tuple[dict, list]:
    history = []
    num_samples = dataset.shape[0]
    
    for step in range(num_steps):
        step_seed = seed + step
        
        # Seed generator prior to sampling minibatch indices
        torch.manual_seed(step_seed)
        indices = torch.randint(0, num_samples, (batch_size,), device=dataset.device)
        x0 = dataset[indices]
        
        # Execute single SGD update step
        params, loss = ddpm_train_step(params, x0, schedule, lr=lr, seed=step_seed)
        history.append(loss)
        
    return params, history

# Step 15 - predict_x0_from_eps
import torch
import torch.nn.functional as F

def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # TODO: invert the q_sample equation for x0
    alphabar = extract_into_batch(alphas_cumprod, t, x_t)
    x0_hat = (x_t - torch.sqrt(1-alphabar)*eps)/torch.sqrt(alphabar)
    return x0_hat

# Step 16 - ddpm_p_mean_variance
import torch
import torch.nn.functional as F

def ddpm_p_mean_variance(x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, schedule: dict):
    alphas = schedule["alphas"]
    alphas_cumprod = schedule["alphas_cumprod"]
    betas = schedule["betas"]
    
    # 1. Estimate clean image x0 from noise prediction eps and clamp to [-1, 1]
    x0_hat = predict_x0_from_eps(x_t, t, eps, alphas_cumprod).clamp(-1.0, 1.0)
    
    # 2. Extract schedule coefficients for step t
    alpha_t = extract_into_batch(alphas, t, x_t.shape)
    beta_t = extract_into_batch(betas, t, x_t.shape)
    alpha_cumprod_t = extract_into_batch(alphas_cumprod, t, x_t.shape)
    
    # 3. Compute alpha_cumprod_{t-1} handling t == 0 boundary condition (alpha_cumprod_{-1} = 1.0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    alpha_cumprod_t_prev = extract_into_batch(alphas_cumprod_prev, t, x_t.shape)
    
    # 4. Compute posterior mean mu_t(x_t, x0_hat)
    coef_x0 = torch.sqrt(alpha_cumprod_t_prev) * beta_t / (1.0 - alpha_cumprod_t)
    coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t)
    mean = coef_x0 * x0_hat + coef_xt * x_t
    
    # 5. Fixed variance choice sigma^2_t = beta_t
    variance = beta_t
    
    return mean, variance, x0_hat

# Step 17 - ddpm_p_sample
import torch
import torch.nn.functional as F

def ddpm_p_sample(x_t: torch.Tensor, t: torch.Tensor, params: dict, schedule: dict, noise: torch.Tensor = None) -> torch.Tensor:
    # 1. Predict noise using the trained tiny UNet
    eps = tiny_unet_forward(x_t, t, params)
    
    # 2. Compute reverse posterior mean and variance
    mean, var, _ = ddpm_p_mean_variance(x_t, t, eps, schedule)
    
    # 3. Sample Gaussian noise if not provided
    if noise is None:
        noise = torch.randn_like(x_t)
        
    # 4. Zero out noise at step t = 0 (t == 0 mask shaped (B, 1, 1, 1))
    nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
    
    # 5. Reverse step sampling: x_{t-1} = mean + sqrt(var) * noise (for t > 0)
    x_prev = mean + nonzero_mask * torch.sqrt(var) * noise
    
    return x_prev

# Step 18 - ddpm_sample_loop
import torch

def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0) -> torch.Tensor:
    # 1. Seed RNG and initialize x from standard normal distribution N(0, I)
    torch.manual_seed(seed)
    x = torch.randn(shape)
    
    T = schedule["T"]
    B = shape[0]
    
    # Determine execution device based on model weights
    device = next(iter(params.values())).device
    x = x.to(device)
    
    # 2. Iterate backward through timesteps T-1 down to 0
    for t in reversed(range(T)):
        # Create a 1D tensor of shape (B,) filled with current timestep t
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        
        # Perform single reverse step: x_{t} -> x_{t-1}
        x = ddpm_p_sample(x, t_batch, params, schedule)
        
    return x

# Step 19 - sample_quality_mse
import torch

def sample_quality_mse(samples: torch.Tensor, dataset: torch.Tensor) -> float:
    """
    Compute the average minimum mean-squared error (MSE) across all samples
    relative to the closest matching image in the reference dataset.
    """
    min_mses = []
    
    for sample in samples:
        # Vectorized MSE across all dataset images for the current sample
        # (sample - dataset) shape: (N_dataset, C, H, W)
        mses = ((sample - dataset) ** 2).mean(dim=(1, 2, 3))
        
        # Store the minimum MSE distance to any image in the dataset
        min_mses.append(mses.min())
        
    # Mean of the minimum MSEs converted to a Python float
    return float(torch.stack(min_mses).mean().item())

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

