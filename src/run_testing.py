import torch
from evaluate import evaluate_epoch
from data_processing import convert_to_dataloader_train_and_test
from vae import VAE
from train import device
import numpy as np
import matplotlib.pyplot as plt

std = 0.5
mean = 0.5

def generate_images(num_images):
    """"
    Generates new Pokemon sprites by sampling random latent vectors from N(0,1)
    and decoding them, bypassing the encoder entirely

    Args:
        num_images: int - number of sprites to generate
    
    Returns:
        new_images: numpy array of shape (num_images, 96, 96, 3), pixel values in [0,1]
    """
    model = VAE()
    model.load_state_dict(torch.load('model_vae_weights.pth'))
    model.to(device)
    model.eval()

    with torch.no_grad():

        z = torch.randn(num_images, 128,device=device)
        new_images = model.decoder(z)

    new_images = new_images*std + mean
    new_images = new_images.permute(0,2,3,1)
    new_images = new_images.cpu().numpy()
    return   new_images


def test_model(test_dataloader):
    """
    Loads the saved model checkpoint and computes reconstruction + KL loss
    on a held-out Dataloader (intended for the test set, touched once).

    Args:
        test_dataloader: Dataloader yielding batches of images, shape (batch, 3, 96, 96)
    
    Returns:
        float - average per-image loss (reconstructiin + KL) on the given DataLoader
    """
    model = VAE()
    model.load_state_dict(torch.load('model_vae_weights.pth'))
    model.to(device)

    Loss_test = (evaluate_epoch(model, test_dataloader))

    return Loss_test

if __name__ == "__main__":
    _, _, test_data = convert_to_dataloader_train_and_test(64)

    Loss_test = test_model(test_data)
    print(f"Loss test is: {Loss_test:.4f}")

    new_images = generate_images(8)
    fig, ax = plt.subplots(2,4)
    for i in range(8):
        ax[i//4, i%4].imshow(new_images[i])

    plt.show()
