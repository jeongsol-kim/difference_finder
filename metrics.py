from typing import Callable
import torch
import lpips
from pytorch_msssim import ssim, ms_ssim

METRICS = {}

def register_metric(name: str) -> Callable:
    def wrapper(fn: Callable) -> Callable:
        if METRICS.get(name) is not None:
            raise NameError(f"Metric {name} is already registered.")
        METRICS[name] = fn
        return fn
    return wrapper

def get_metric(name: str) -> Callable:
    if METRICS.get(name) is None:
        raise NameError(f"Metric {name} does not exist.")
    return METRICS[name]


@register_metric(name='mse')
def MeanSquaredError(img1: torch.Tensor,
                     img2: torch.Tensor) -> torch.Tensor:
    mse = torch.nn.MSELoss()
    return mse(img1, img2) 

@register_metric(name='lpips')
def PerceptualLoss(img1: torch.Tensor,
                   img2: torch.Tensor) -> torch.Tensor:
    fn = lpips.LPIPS(net='vgg').to(img1.device).to(img1.dtype)
    return fn(img1, img2)

@register_metric(name='ssim')
def StructureSimilarity(img1: torch.Tensor,
                        img2: torch.Tensor):
    # TODO: how to handle data_range & win_size
    return ssim(img1, img2)

@register_metric(name='ms-ssim')
def MultiScaleStructureSimilarity(img1: torch.Tensor,
                                  img2: torch.Tensor):
    return ms_ssim(img1, img2) 

@register_metric(name='psnr')
def PeakSignalToNoiseRatio(img1: torch.Tensor,
                           img2: torch.Tensor):
    mse = torch.mean((img1-img2)**2)
    return 20 * torch.log10(1.0/torch.sqrt(mse))