import torch
# from torchvision.datasets import MNIST
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from datasets.celeba2ganimage import CelebA2GANImage
from models.unsupervised.gan.small_model import Generator, Discriminator

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

batch_size = 32
latent_dim = 100
epochs = 100
lr = 2e-4

transform = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
])

# mnist_dataset = MNIST(root="data", train=True, transform=transform, download=True)

dataset = CelebA2GANImage(root_dir="../../../data/celeba/img_align_celeba/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size)

generator = Generator(latent_dim=latent_dim)
discriminator = Discriminator()
generator.to(device)
discriminator.to(device)

gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=lr)
dis_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr)

for epoch in range(epochs):
    d_running_loss, g_running_loss = 0.0, 0.0

    for real_img, _ in dataloader:
        real_img = real_img.to(device)

        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()

        ## Generator
        z = torch.randn(real_img.shape[0], latent_dim).to(device)
        fake_imgs = generator(z)
        verdict = discriminator(fake_imgs)

        g_loss = F.binary_cross_entropy(verdict, torch.ones(real_img.shape[0], 1).to(device))
        g_loss.backward()
        gen_optimizer.step()

        ## Discriminator
        verdict = discriminator(real_img)
        real_loss = F.binary_cross_entropy(verdict, torch.ones(real_img.shape[0], 1).to(device))

        verdict = discriminator(generator(z).detach())
        fake_loss = F.binary_cross_entropy(verdict, torch.zeros(real_img.shape[0], 1).to(device))

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        dis_optimizer.step()

        g_running_loss += g_loss / real_img.shape[0]
        d_running_loss += d_loss / real_img.shape[0]

    print(f"Epoch {epoch} \n\tDiscriminator loss: {g_running_loss / len(mnist_dataloader)} "
          f"\n\tGenerator loss: {d_running_loss / len(mnist_dataloader)}")
