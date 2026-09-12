from explore_data import build_records
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset ,random_split
from PIL import Image
import torch

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

records = build_records()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5))
])


dataset_records = ImageDataSet(records,transform)

dataset_records_train, dataset_records_test = random_split(dataset_records, [0.9, 0.1])
print("Shape of first dataset_records_train: ", dataset_records_train[0].shape)
print("max and min value of first sample in dataset_records_train: ", dataset_records_train[0].max(), dataset_records_train[0].min())

data_loader_train = DataLoader(dataset_records_train, batch_size= 64, shuffle= True)
data_loader_test = DataLoader(dataset_records_test, batch_size=64)

