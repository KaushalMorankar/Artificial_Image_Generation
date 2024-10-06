import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.view(-1, 784))

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=64, shuffle=True)

# Initialize models
netG = Generator().cuda()
netD = Discriminator().cuda()

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002)

num_epochs = 200

for epoch in range(num_epochs):
    for i, data in enumerate(mnist_loader, 0):
        # Train Discriminator
        netD.zero_grad()
        real, _ = data
        real = real.cuda()
        batch_size = real.size(0)
        labels = torch.full((batch_size,), 1, dtype=torch.float, device='cuda')
        
        output = netD(real).view(-1)
        lossD_real = criterion(output, labels)
        lossD_real.backward()

        noise = torch.randn(batch_size, 100, device='cuda')
        fake = netG(noise)
        labels.fill_(0)

        output = netD(fake.detach()).view(-1)
        lossD_fake = criterion(output, labels)
        lossD_fake.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        labels.fill_(1)

        output = netD(fake).view(-1)
        lossG = criterion(output, labels)
        lossG.backward()
        optimizerG.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss D: {lossD_real + lossD_fake}, Loss G: {lossG}')

# Generate a new image using the trained Generator
noise = torch.randn(1, 100, device='cuda')
generated_image = netG(noise).detach().cpu().numpy()
generated_image = (generated_image + 1) / 2  # Rescale from [-1, 1] to [0, 1]

# Display the generated image
plt.imshow(generated_image.squeeze(), cmap='gray')
plt.title('Generated Image')
plt.show()
