import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Data loader
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# desired variance in noise (can be changed freely)
sigma = 64
d = 2

# Adding Gaussian noise
def add_noise(images, sigma=1): #sigma is our desired variance in our normal distribution
    sigma /= 255
    noisy_imgs = images + (sigma ** 0.5) * torch.randn(*images.shape) # multipying by sqrt(sigma) to get desired variance in noise
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    return noisy_imgs

# Basic CNN model for denoising
class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Output: (32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: (64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: (64, 7, 7)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # Output: (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2), # Output: (1, 28, 28)
            nn.Sigmoid() # Ensuring output is between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = DenoisingCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        noisy_images = add_noise(images).to(device)
        images = images.to(device)

        # formulas presented in the paper for reconstruction (d chosen arbitrarily to be 2)
        recorrupt_input = noisy_images + d*((sigma / 255) ** 0.5)*torch.FloatTensor(noisy_images.size()).normal_(mean=0,std=1).to(device) # Introducing noise to our noisy image (in this case we will add 64 ? )
        recorrupt_target = noisy_images - (1/d)*((sigma / 255) ** 0.5)*torch.FloatTensor(noisy_images.size()).normal_(mean=0,std=1).to(device)

        outputs = model(recorrupt_input)
        loss = criterion(outputs, recorrupt_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print('Finished Training')

import matplotlib.pyplot as plt

def visualize_denoising(model, device, test_loader, num_images=5):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        dataiter = iter(test_loader)
        images, _ = next(dataiter)

        # Add noise to the images and move them to the configured device
        noisy_images = add_noise(images, sigma).to(device)

        # Get the model's predictions (denoised images)
        denoised_images = model(noisy_images)

        # Move images back to CPU for visualization
        noisy_images = noisy_images.cpu()
        denoised_images = denoised_images.cpu()

        # Plot the results
        fig, axs = plt.subplots(num_images, 3, figsize=(10, num_images * 3))
        for i in range(num_images):
            axs[i, 0].imshow(noisy_images[i].squeeze(), cmap='gray')
            axs[i, 0].title.set_text('Noisy Image')
            axs[i, 1].imshow(denoised_images[i].squeeze(), cmap='gray')
            axs[i, 1].title.set_text('Denoised Image')
            axs[i, 2].imshow(images[i].squeeze(), cmap='gray')
            axs[i, 2].title.set_text('Original Image')
            diff = torch.sum(torch.square(denoised_images[i] - images[i]))
            print("Losses by Image: " + str(diff.item()))
            for ax in axs[i]:
                ax.axis('off')
        plt.tight_layout()
        plt.show()

# Call the visualization function
visualize_denoising(model, device, test_loader, num_images=5)