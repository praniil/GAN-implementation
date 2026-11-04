import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import torch.nn as nn

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
latent_dim = 100
EPOCHS = 50
IMAGE_SIZE = 784

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("../generated", exist_ok=True)
print("using device: ", DEVICE)


# generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, z):
        # self.model(z) -> in shape [x, 784] -> x is no of images
        image = self.model(z)
        # image shape -> [no of image, 784]
        image = image.view(-1, 1, 28, 28) # -1 tells automaitcally calculate the dimension of the remaining dimensions and toatl no of elements
        # if 4 images then (4 * 74) / (1 * 28 * 28) = 4  => [4, 1, 28, 28]

        # image shape -> [1, 28, 28]
        return image


# descriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(IMAGE_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # flatening the image
        # [32, 1, 28, 28] -> [32, 784]
        img = img.view(-1, IMAGE_SIZE)
        validity = self.model(img)
        return validity

# initialize networks
generator = Generator(latent_dim).to(DEVICE)
descriminator = Discriminator().to(DEVICE)

