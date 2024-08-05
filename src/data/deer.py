import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt


class DeerDataset(Dataset):
    def __init__(self, cifar10_dataset):
        self.data = []
        self.targets = []
        
        for i in range(len(cifar10_dataset)):
            img, label = cifar10_dataset[i]
            if label == 4:  # 4 is the label for deer in CIFAR-10
                self.data.append(img)
                self.targets.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def display_deer_image(dataset, index=None):
    if index is None:
        index = np.random.randint(len(dataset))
    
    image, label = dataset[index]
    
    # Convert the image from PyTorch tensor to numpy array
    image = image.numpy()
    
    # Denormalize the image
    image = image / 2 + 0.5  # reverse the normalization
    image = np.transpose(image, (1, 2, 0))  # change from (C, H, W) to (H, W, C)
    print(image.shape)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Deer Image (Index: {index})")
    plt.axis('off')
    plt.show()


def get_deer_dataloaders(batch_size=32, val_split=0.2, num_workers=2, random_seed=42):
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load CIFAR-10 dataset
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=transform)
    
    # Create DeerDataset
    deer_dataset = DeerDataset(cifar10_dataset)
    
    # Calculate sizes for train and validation sets
    dataset_size = len(deer_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(deer_dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
    
    return train_dataloader, val_dataloader
    

if __name__ == "__main__":
    # Get the deer dataloader
    deer_dataloader = get_deer_dataloaders()
    
    # Print some information about the dataset
    
    # Iterate through the dataloader to verify it's working
    for i, (images, labels) in enumerate(deer_dataloader):
        if i == 0:  # Print info for the first batch only
            break