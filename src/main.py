import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

transform = transforms.Compose([
    #.ToTensor() -> image to pytorch tensor
    # pixels transformed from 0 to 1
    transforms.ToTensor(),
])

# train dataset
train_dataset = torchvision.datasets.MNIST(
    root="../dataset",
    train=True,
    transform=transform,
    download=True
)

# information about the first mnist image
print("Dataset size: ", len(train_dataset))
image, label = train_dataset[0]
print("Image shape: ", image.shape)
print("label: ", label)
print("Minimum pixel value: ", image.min)
print("Maximum pixel value: ", image.max)

# visulaize the image
# image.squeeze() removes the dimension of size 1
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Lable: {label}")
plt.axis("off")
plt.savefig("../saved/mnist-data.jpg")

# load data
batch_size = 64

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# inspect one batch
images, labels = next(iter(train_loader))
print("batch shape: ", images.shape)
print("labels shape: ", labels.shape)

# training configuration