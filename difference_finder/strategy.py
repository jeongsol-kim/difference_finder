from typing import Callable, Optional
import torch

STRATEGY = {}

def register_strategy(name: str) -> Callable:
    def wrapper(fn: Callable) -> Callable:
        if STRATEGY.get(name) is not None:
            raise NameError(f"Strategy {name} is already registered.")
        STRATEGY[name] = fn
        return fn
    return wrapper
    

def get_strategy(name: str):
    if STRATEGY.get(name) is None:
        raise NameError(f"Strategy {name} does not exist.")
    return STRATEGY[name]

@register_strategy(name='difference')
def difference_strategy(img1: torch.Tensor,
                        img2: torch.Tensor,
                        metric_fn: Optional[Callable]=None):
    metric_fn = lambda x, y: torch.abs(x-y)
    difference = metric_fn(img1, img2)
    assert difference.shape == img1.shape, \
        print('It is expected that metric_fn returns the same dimentional tensor as the input.')
    
    return difference

@register_strategy(name='gradient')
def gradient_strategy(img1: torch.Tensor,
                      img2: torch.Tensor,
                      metric_fn: Callable,
                      reference_idx: Optional[int]=0):

    assert reference_idx in [0, 1], \
        print('reference_idx should be 0 or 1 (i.e. img1 or img2).')
    
    target = img1 if reference_idx==0 else img2 
    compare = img2 if reference_idx==0 else img1

    target.requires_grad = True
    scalar = metric_fn(compare, target).mean()
    diff_grad = torch.autograd.grad(scalar, target)[0]
    return diff_grad