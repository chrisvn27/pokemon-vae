import torch
import torch.nn as nn
from train import kl_divergence_loss, device, calculate_weighted_error

def evaluate_epoch(model, data_test_or_val):
    """
    Runs one epoch of evaluation: forward pass computes the combined reconstruction + KL loss
    without annealing to keep the true losses for every batch in data_test_or_val

    Args:
        model: VAE instance
        data_test_or_val: DataLoader yielding batches of images, shape (batch, 3, 96, 96)

    Returns:
        float - average per image-loss (reconstruction + KL) for this epoch
    """
    model.eval()
    beta = 0.5
    Loss_track = 0
    with torch.no_grad():
        for x in data_test_or_val:
            x = x.to(device)
            mu, log_var, x_recon = model(x)
            weighted_error = calculate_weighted_error(x, x_recon)
            recon_loss = weighted_error.sum()
            Loss =  recon_loss + beta * kl_divergence_loss(mu, log_var)
            Loss_track += Loss.item()/len(x)

    return Loss_track/len(data_test_or_val)



    
