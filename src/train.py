import torch 
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
file_name_pth = "model_vae_weights_250epochs_kl50percent_0p5beta_weightedmask.pth"

def kl_divergence_loss(mu, log_var):
    """
    Computes the closed-form KL divergence between N(mu, exp(log_var)) and N(0, 1),
    summed over latent dimensions and summed over the batch.

    Args:
        mu: tensor of shape (batch, 128)
        log_var: tensor of shape (batch, 128)
    Returns:
        scalar tensor - total KL divergence across the batch
    """
    kl = -0.5*(1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)
    return kl.sum(dim=0)

def calculate_weighted_error(x,x_recon):
    """
    Computes squared reconstruction error weighted per-pixel, giving foreground
    (non-white) pixels 5x the weight of background (white) pixels, so foreground
    detail isn't drowned out by the large uniform background region.

    Args:
        x: tensor of shape (batch, 3, 96, 96) - original images, pixel values in [-1, 1]
        x_recon: tensor of shape (batch, 3, 96, 96) - reconstructed images

    Returns:
        tensor of shape (batch, 3, 96, 96) - weighted squared error, not yet summed
    """
    # Exact equality is safe here: sprites are palette/flat-color PNGs with no
    # anti-aliasing, so background pixels are reliably exactly 1.0 (verified:
    # (x==1).sum() == (x>0.99).sum() on a full batch)    
    is_white = (x == 1)
    background_mask = torch.all(is_white, dim=1)

    weight = torch.where(background_mask, 1.0, 5.0)
    weight = weight.unsqueeze(1)

    squared_error = (x - x_recon) ** 2
    weighted_error = squared_error*weight

    return weighted_error



def train_epoch(model, optim, data_train, epoch, epochs):
    """
    Runs one epoch of training: forward pass, combined weighted-reconstruction + KL loss
    (KL annealed linearly from 0 to a cap of 0.5 over the first 50% of epochs), backward
    pass, and optimizer step, for every batch in data_train.

    Args:
        model: VAE instance
        optim: optimizer (e.g. torch.optim.Adam) over model's parameters
        data_train: DataLoader yielding batches of images, shape (batch, 3, 96, 96)
        epoch: current epoch number (0-indexed), used to compute the KL annealing weight
        epochs: total number of epochs training will run for
    Returns:
        float - average per-image loss (weighted reconstruction + beta*KL) for this epoch
    """
    model.train()
    beta = min(0.5, epoch / max(1, int(0.5*epochs)))
    Loss_track = 0
    for x in data_train:
        x = x.to(device)
        optim.zero_grad()
        mu, log_var, x_recon = model(x)
        weighted_error = calculate_weighted_error(x, x_recon)
        recon_loss = weighted_error.sum()
        Loss = recon_loss + beta*kl_divergence_loss(mu, log_var)
        Loss.backward()
        optim.step()

        Loss_track += Loss.item() / len(x)

    return Loss_track / len(data_train)

        

