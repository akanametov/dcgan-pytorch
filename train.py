import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import time
import argparse
from progress.bar import IncrementalBar

from dataset import CelebA, LSUN
from generator import Generator
from discriminator import Discriminator
from utils import initialize_weights

parser = argparse.ArgumentParser(prog = 'top', description='Train DCGAN')
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="celeba", help="Name of the dataset: ['celeba', 'lsun']")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
args = parser.parse_args()

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

transforms = T.Compose([
    T.Resize((64, 64)),
    T.CenterCrop((64, 64)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))
])

print('Defining models...')
generator = Generator().to(device)
discriminator = Discriminator().to(device)
# optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
# loss functions
g_criterion = nn.BCELoss().to(device)
d_criterion = nn.BCELoss().to(device)
# dataset
print(f'Downloading "{args.dataset.upper()}" dataset ...')
if args.dataset=='celeba':
    dataset = CelebA(root='.', download=True, transform=transforms)
elif args.dataset=='lsun':
    dataset = LSUN(root='.', download=True, transform=transforms)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
# train
print("Start Training...")
min_loss = math.inf
for epoch in range(args.epochs):
    de_loss=0.
    ge_loss=0.
    start = time.time()
    bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]', max=len(dataloader))
    for real, _ in dataloader:
        real = real.to(device) 
        # Train Discriminator
        discriminator.zero_grad()
        # real loss
        real_pred = discriminator(real)
        real_loss = d_criterion(real_pred, torch.ones_like(real_pred))
        real_loss.backward()
        # generate fake image
        noise = torch.randn(len(real), 100, 1, 1).to(device)
        fake = generator(noise)
        # fake loss
        fake_pred = discriminator(fake.detach())
        fake_loss = d_criterion(fake_pred, torch.zeros_like(fake_pred))
        fake_loss.backward()
        # discriminator loss
        d_loss = real_loss + fake_loss
        # update parameters
        d_optimizer.step()
        
        # Train Generator
        generator.zero_grad()
        # fooling loss
        fake_pred = discriminator(fake)
        g_loss = g_criterion(fake_pred, torch.ones_like(fake_pred))
        g_loss.backward()
        # update parameters
        g_optimizer.step()
        
        de_loss += d_loss.item()
        ge_loss += g_loss.item()
        bar.next()
    bar.finish()  
    g_loss = ge_loss/len(dataloader)
    d_loss = de_loss/len(dataloader)
    end = time.time()
    tm = (end - start)
    if min_loss > g_loss:
        min_loss = g_loss
        torch.save(generator.state_dict(), 'generator_celeba.pt')
        torch.save(discriminator.state_dict(), 'discriminator_celeba.pt')
    print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch+1, args.epochs, g_loss, d_loss, tm))
