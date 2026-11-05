import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch.optim as optim

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
LEARNING_RATE = 0.001

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
discriminator = Discriminator().to(DEVICE)

# loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

# fixed noise for visulationzation
# 64 -> number of noise vectors, 100 = number of values in each noise vector
# they are the points in the latent space
fixed_noise = torch.randn(
    64, latent_dim, device=DEVICE
)

# training loop
for epoch in range(EPOCHS):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(DEVICE)
        current_batch_size = real_images.size(0)
        real_labels = torch.ones(
            current_batch_size, 1, device=DEVICE
        )
        fake_labels = torch.zeros(
            current_batch_size, 1, device=DEVICE
        )

        # train descriminator
        optimizer_D.zero_grad()

        # real images
        real_output = discriminator(real_images)
        loss_real = criterion(
            real_output, real_labels
        )

        # fake images
        noise = torch.randn(
            current_batch_size, 
            latent_dim,
            device=DEVICE
        )

        fake_images = generator(noise)
        fake_output = discriminator(fake_images.detach())

        loss_fake = criterion(
            fake_output, fake_labels
        )

        # Total discriminator loss
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # train generator
        optimizer_G.zero_grad()

        # generate new images
        noise = torch.randn(
            current_batch_size,
            latent_dim, 
            device=DEVICE
        )

        generated_images = generator(noise)

        # generator wants discriminator to classify genearted images as real

        output = discriminator(generated_images)
        loss_G = criterion(
            output, real_labels
        )

        loss_G.backward()
        optimizer_G.step()

        print(
            f"Epoch[{epoch + 1}/ {EPOCHS}]"
            f"Loss D: {loss_D.item(): .4f}"
            f"Loss G: {loss_G.item(): .4f}"
        )
        

