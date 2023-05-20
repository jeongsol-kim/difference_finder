from pathlib import Path
from typing import Optional, Callable, List
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImagePathDataset(Dataset):
    def __init__(self, files, transforms=None) -> None:
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        path = self.files[index]
        img = Image.open(path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
        return img
        
        
class TwoImagePathDataset(Dataset):
    def __init__(self, files1, files2, transforms=None) -> None:
        self.files1 = files1
        self.files2 = files2
        self.transforms = transforms

        assert len(self.files1) == len(self.files2), \
            f'Two file lists should have the same number of files. \
                Got {len(self.files1)} and {len(self.files2)}.'

    def __len__(self):
        return len(self.files1)
    
    def __getitem__(self, index):
        path1 = self.files1[index]
        path2 = self.files2[index]

        img1 = Image.open(path1).convert('RGB')
        img2 = Image.open(path2).convert('RGB')

        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        return img1, img2


def get_loader(files1: List[Path],
               files2: List[Path],
               transform: Optional[Callable]=None,
               num_workers: Optional[int]=0) -> DataLoader:
    
    # read all files (png, jpg, jpeg)

    dataset = TwoImagePathDataset(files1, files2, transform)
    return DataLoader(dataset, num_workers=num_workers)