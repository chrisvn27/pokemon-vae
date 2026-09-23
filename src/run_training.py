import torch
from train import train_epoch, device
from evaluate import evaluate_epoch
from vae import VAE
from data_processing import convert_to_dataloader_train_and_test
import matplotlib.pyplot as plt

def train_model(model, batch_size, optim, epochs):
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
            print(f"Epoch {e}: Train Loss: {Loss_train[e]:.4f}, Val Loss: {Loss_val[e]:.4f}")

    return Loss_train, Loss_val


if __name__ == "__main__":
    epochs = 2

    model = VAE()
    optim = torch.optim.Adam(model.parameters())
    batch_size = 64

    loss_train, loss_val = train_model(model,batch_size, optim, epochs)

    plt.plot(loss_train, label="Loss train")
    plt.plot(loss_val, label = "Loss val")
    plt.legend()
    plt.show()
    

