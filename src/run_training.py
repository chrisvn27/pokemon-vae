import torch
from train import train_epoch, device
from evaluate import evaluate_epoch
from vae import VAE
from data_processing import convert_to_dataloader_train_and_test
import matplotlib.pyplot as plt

def train_model(model, batch_size, optim, epochs):
    """
    Trains a VAE for a fixed number of epochs, tracking training and validation
    loss each epoch, saves the model's weights to 'model_vae_weights.pth'
    whenever validation loss reaches a new best value.

    Args:
        model: VAE instance (moved to device internally)
        batch_size: int - batch size used for the train/val/test DataLoaders
        optim: optimizer (e.g. torch.optim.Adam) over model's parameters
        epochs: int - number of epochs to train for

    Returns:
        Loss_train: list of float, length epochs - average per image training loss per epoch
        Loss_val : list of float, length epochs, average per-image validation loss per epoch

    """
    train_dataloader, val_dataloader, _ = convert_to_dataloader_train_and_test(batch_size)

    Loss_train = []
    Loss_val  = []

    best_recorded = torch.inf

    model.to(device)

    for e in range(epochs):
        Loss_train.append(train_epoch(model, optim, train_dataloader, e, epochs))
        Loss_val.append(evaluate_epoch(model,val_dataloader))

        if Loss_val[-1] < best_recorded:
            torch.save(model.state_dict(), 'model_vae_weights.pth')
            best_recorded = Loss_val[-1]

        if (e+1) % 25 == 0 :
            print(f"Epoch {e+1}: Train Loss: {Loss_train[e]:.4f}, Val Loss: {Loss_val[e]:.4f}")

    return Loss_train, Loss_val


if __name__ == "__main__":
    epochs = 50

    model = VAE()
    optim = torch.optim.Adam(model.parameters())
    batch_size = 64

    loss_train, loss_val = train_model(model,batch_size, optim, epochs)

    plt.plot(loss_train, label="Loss train")
    plt.plot(loss_val, label = "Loss val")
    plt.legend()
    plt.show()
    

