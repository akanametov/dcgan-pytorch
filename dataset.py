import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import download_and_extract

class CelebA(Dataset):
    """CelebA dataset."""
    url="https://github.com/akanametov/dcgan/releases/download/1.0/celeba.zip"
    def __init__(self, root, download=False, transform=None):
        if download:
            _ = download_and_extract(root, self.url)
        self.root=root
        self.files=sorted(glob.glob(f"{root}/celeba/img_align_celeba/*.jpg"))
        self.transform=transform
        self.download=download
        
    def __len__(self,):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([0]).long()
    
    
class LSUN(Dataset):
    """LSUN(bedroom) dataset."""
    url="https://github.com/akanametov/dcgan/releases/download/1.0/lsun.zip"
    def __init__(self, root, download=False, transform=None):
        if download:
            _ = download_and_extract(root, self.url)
        self.root=root
        self.files=sorted(glob.glob(f"{root}/lsun/bedroom/0/*/*/*.jpg"))
        self.transform=transform
        self.download=download
        
    def __len__(self,):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([0]).long()
