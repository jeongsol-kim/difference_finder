from typing import Optional
import torch
from difference_finder.metrics import get_metric
from difference_finder.strategy import get_strategy
from difference_finder.processor import get_preprocessor, get_postprocessor

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

    def run(self, img1:torch.Tensor, img2:torch.Tensor):
        img1 = self.pre_processor(img1)
        img2 = self.pre_processor(img2)
        _map = self.strategy(img1, img2, metric_fn=self.metric)
        output = self.post_processor(_map)
        return output

    def save_fn(self):
        pass

        
        
        
# class Finder(object):
#     def __init__(self,
#                  file1: Union[List[Path], Path],
#                  file2: Union[List[Path], Path],
#                  num_workers: Optional[int]=0,
#                  preprocess_fn: Optional[Callable]=None,
#                  postprocess_fn: Optional[Callable]=None):

#         if file1.is_dir() and file2.is_dir():
#             # if given files is a directory, retrieve all images (png, jpg, jpeg)
#             glob_fn = lambda x: list(x.glob('*.png')) \
#                                 + list(x.glob('*.jpg')) \
#                                 + list(x.glob('*.jpeg'))
#             files1 = sorted(glob_fn(file1))
#             files2 = sorted(glob_fn(file2))
#         else:
#             # else, assume given files is a list of paths
#             files1 = file1
#             files2 = file2
    
#         assert len(files1) == len(files2) 
#         self.loader = get_loader(files1,
#                                  files2,
#                                  transform=ToTensor(),
#                                  num_workers=num_workers)
        
#         if preprocess_fn is not None:
#             self.preprocess = preprocess_fn
#         else:
#             self.preprocess = lambda x: x
        
#         if postprocess_fn is not None:
#             self.postprocess = postprocess_fn
#         else:
#             self.postprocess = lambda x: x

#     def find(self):
#         strategy = get_strategy('threshold')
#         metric = get_metric('ssim')
     
#     def run(self):
#         for (img_1, img_2) in self.loader:
#             img_1 = self.preprocess(img_1)
#             img_2 = self.preprocess(img_2)
#             print(img_1.shape, img_2.shape)

            
