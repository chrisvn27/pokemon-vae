import torch 
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_recon = nn.MSELoss(reduction='sum')

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

def train_epoch(model, optim, data_train, epoch, epochs):
    """
    Runs one epoch of training: forward pass, combined reconstruction + KL loss
    (with linear KL annealing over the first 20% of epochs), backward pass, and
    optimizer step, for every batch in data_train.

    Args:
        model: VAE instance
        optim: optimizer (e.g. torch.optim.Adam) over model's parameters
        data_train: DataLoader yielding batches of images, shape (batch, 3, 96, 96)
        epoch: current epoch number (0-indexed), used to compute the KL annealing weight
        epochs: total number of epochs training will run for
    Returns:
        float - average per-image loss (reconstruction + beta*KL) for this epoch
    """
    model.train()
    beta = min(1, epoch / max(1, int(0.2*epochs)))
    Loss_track = 0
    for x in data_train:
        x = x.to(device)
        optim.zero_grad()
        mu, log_var, x_recon = model(x)
        Loss = loss_recon(x_recon, x) + beta*kl_divergence_loss(mu, log_var)
        Loss.backward()
        optim.step()

        Loss_track += Loss.item() / len(x)

    return Loss_track / len(data_train)

        

