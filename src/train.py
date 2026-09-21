import torch 
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_recon = nn.MSELoss(reduction='sum')

def kl_divergence_loss(mu, log_var):
    kl = -0.5*(1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)
    return kl.sum(dim=0)

def train(model, betha, optim, data_train):
    model.train()
    for x in data_train:
        x = x.to(device)
        optim.zero_grad()
        mu, log_var, x_recon = model(x)
        Loss = loss_recon(x_recon, x) + betha*kl_divergence_loss(mu, log_var)
        Loss.backward()
        optim.step()
        

