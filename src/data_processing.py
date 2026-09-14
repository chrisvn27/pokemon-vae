from explore_data import build_records
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset ,random_split
from PIL import Image
import torch

torch.manual_seed(42)

class ImageDataSet(Dataset):
    def __init__(self,records, transform):
        super().__init__()
        self.records = records
        self.transform = transform


    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img = Image.open(record["path"])
        img = self.transform(img)
        return img

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5))
])


def convert_to_dataloader_train_and_test(batch_size = batch_size,transform = transform):
    records = build_records()
    dataset_records = ImageDataSet(records,transform)
    dataset_records_train, dataset_records_test = random_split(dataset_records, [0.9, 0.1])
    dataloader_train = DataLoader(dataset_records_train, batch_size= batch_size, shuffle= True)
    dataloader_test = DataLoader(dataset_records_test, batch_size=batch_size)

    return dataloader_train, dataloader_test



if __name__ == "__main__":

    train_dataloader , test_dataloader = convert_to_dataloader_train_and_test()

    first_batch_check = next(iter(train_dataloader))
    print("Number of images per batch: ", {len(first_batch_check)})
    print("Shape of first dataset_records_train: ", first_batch_check.shape)
    print("max and min value of first sample in dataset_records_train: ", first_batch_check[0].max(), first_batch_check[0].min())






