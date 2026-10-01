from run_training import train_model
from run_testing import generate_images, test_model, generate_images_fromdataloader
from vae import VAE
from data_processing import convert_to_dataloader_train_and_test
import matplotlib.pyplot as plt
import torch
import time


if __name__ == "__main__":
    answer = ''
    keys = ['y','n','Y','N']
    while answer not in keys:
        answer = input("Run train? (y/n) (don't forget to update "
                       "weight_parameters name in train.py accordingly when testing): ")

    epochs = 250
    batch_size = 64

    num_images = 4
    

    if answer == 'y':
        start_time = time.perf_counter()
        model = VAE()
        optim = torch.optim.Adam(model.parameters())

        loss_train, loss_val = train_model(model, batch_size, optim, epochs)

        end_time = time.perf_counter()
        delta_time = end_time - start_time

        print(f"Elapsed training time: {delta_time:.2f} seconds")

        plt.figure()
        plt.plot(loss_train, label="Train Loss")
        plt.plot(loss_val, label="Validation Loss")
        plt.yscale('log')
        plt.legend()

    train_dataloader, _, test_dataloader = convert_to_dataloader_train_and_test(batch_size)

    loss_test = test_model(test_dataloader)
    print(f"Loss test is: {loss_test:.4f}")

    

    fig, ax = plt.subplots(3,4)

    gen_img, true_img = generate_images_fromdataloader(test_dataloader)
    random_img = generate_images(num_images)

    for i in range(num_images):
        ax[0,i].imshow(gen_img[i])
        ax[1,i].imshow(true_img[i])
        ax[2, i].imshow(random_img[i])
    ax[0,0].set_ylabel("Reconstructed")
    ax[1,0].set_ylabel("Original")
    ax[2,0].set_ylabel("Random")

    gen_img, true_img = generate_images_fromdataloader(train_dataloader)

    fig1, ax1 = plt.subplots(2,4)

    for i in range(num_images):
        ax1[0,i].imshow(gen_img[i])
        ax1[1,i].imshow(true_img[i])
    ax1[0,0].set_ylabel("Reconstructed")
    ax1[1,0].set_ylabel("Original")
    plt.show()


