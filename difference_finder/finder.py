from pathlib import Path
from typing import Optional, Union, List
import torch
from torchvision.transforms import ToTensor
from difference_finder.utils import to_np
from difference_finder.metrics import get_metric
from difference_finder.strategy import get_strategy
from difference_finder.processor import get_preprocessor, get_postprocessor
from difference_finder.data import get_loader

def get_image_files(dir: Path) -> List[Path]:
    return sorted(dir.glob('*.png')) +\
           sorted(dir.glob('*.jpg')) +\
           sorted(dir.glob('*.jpeg'))

class Finder(object):
    def __init__(self,
                 pre_processor: Optional[object]='identity',
                 post_processor: Optional[object]='identity',
                 strategy: Optional[str]='gradient',
                 metric: Optional[str]='mse',
                 ):
        self.pre_processor = get_preprocessor(name=pre_processor)
        self.post_processor = get_postprocessor(name=post_processor)
        self.strategy = get_strategy(name=strategy)
        self.metric = get_metric(name=metric)

    def run_on_image(self, img1: torch.Tensor, img2: torch.Tensor):
        img1 = self.pre_processor(img1)
        img2 = self.pre_processor(img2)
        _map = self.strategy(img1, img2, metric_fn=self.metric)
        output = self.post_processor(_map)

        # output as numpy
        # remove batch dimension and reduce mean.
        # shared for every post_process, so excluded.
        output = to_np(output)[0].mean(axis=-1)
        return output
    
    def run_on_directory(self, img1: Path, img2: Path):
        files1 = get_image_files(img1)
        files2 = get_image_files(img2)
        loader = get_loader(files1=files1,
                            files2=files2,
                            transform=ToTensor(),
                            num_workers=0
                            )
        outputs = []
        for (first_img, second_img) in loader:
            output = self.run_on_image(first_img, second_img)
            outputs.append(output)
        return outputs
    
    def run(self,
            img1: Union[Path, torch.Tensor],
            img2: Union[Path, torch.Tensor]):

        if isinstance(img1, Path) and isinstance(img2, Path):
            if img1.is_dir() and img2.is_dir():
                output = self.run_on_directory(img1, img2)
            else:
                loader = get_loader(files1=[img1],
                                    files2=[img2],
                                    transform=ToTensor(),
                                    num_workers=0)
                first_img, second_img = next(iter(loader))
                output = self.run_on_image(first_img, second_img)
        else:
            output = self.run_on_image(img1, img2)
        return output

    def save_fn(self):
        pass