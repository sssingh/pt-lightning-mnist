from os import cpu_count
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST 
import pytorch_lightning as pl

# All data handling is done in this class
class ImageDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
   
    
    # Download and preprocess (read/write to disk etc)
    def prepare_data(self):
        MNIST(root='data', train=True, download=True)
        MNIST(root='data', train=False, download=True)
    
    
    # Prepare datasets, train, val test splits
    def setup(self, stage=None):
        if stage in ['fit', 'validate']:
            self.train_ds = MNIST(root='data', train=True, download=False,transform=transforms.ToTensor())
            # split train_ds into train (80%) and val (20%) sets
            self.train_ds, self.val_ds = random_split(dataset=self.train_ds,lengths=[48000, 12000])
        if stage in ['test']:            
            self.test_ds = MNIST(root='data', train=False, download=False,transform=transforms.ToTensor())
            
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=32, shuffle=True, num_workers=cpu_count())
  
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=32, shuffle=False, num_workers=cpu_count())
   
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=32, shuffle=False, num_workers=cpu_count())