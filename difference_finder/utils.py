from typing import Optional, Union
import numpy as np
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift

def normalize(img: Union[torch.Tensor, np.ndarray]) \
                        -> Union[torch.Tensor, np.ndarray]:
    
    return (img - img.min())/(img.max()-img.min())
     
def to_np(img: torch.Tensor,
          mode: Optional[str]='NCHW') -> np.ndarray:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        img = img.permute(0,2,3,1) 

    return img.detach().cpu().numpy()

def fft2d(img: torch.Tensor,
          mode: Optional[str]='NCHW') -> torch.Tensor:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        return fftshift(fft2(img))
    elif mode == 'NHWC':
        img = img.permute(0,3,1,2)
        return fftshift(fft2(img))
    else:
        raise NameError    
    

def ifft2d(img: torch.Tensor,
           mode: Optional[str]='NCHW') -> torch.Tensor:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        return ifft2(ifftshift(img))
    elif mode == 'NHWC':
        img = ifft2(ifftshift(img))
        return img.permute(0,2,3,1)
    else:
        raise NameError    


def hp_filter(img: torch.Tensor,
              factor: Optional[float]=0.5) -> torch.Tensor:
    
    # assume img.shape = (N, C, H, W) 
    # shape should be changed before filter

    H, C, H, W = img.shape
    mask = torch.ones_like(img, device=img.device)

    cent_h = int(H//2)
    cent_w = int(W//2)
    half_width_h = int(H*factor//2)
    half_width_w = int(W*factor//2)
    
    mask[:, :, cent_h-half_width_h:cent_h+half_width_h, cent_w-half_width_w:cent_w+half_width_w] = 0
    
    return img * mask 

def lp_filter(img: torch.Tensor,
              factor: Optional[float]=0.5):

    # assume img.shape = (N, C, H, W) 

    H, C, H, W = img.shape
    mask = torch.zeros_like(img, device=img.device)

    cent_h = int(H//2)
    cent_w = int(W//2)
    half_width_h = int(H*factor//2)
    half_width_w = int(W*factor//2)
    
    mask[:, :, cent_h-half_width_h:cent_h+half_width_h, cent_w-half_width_w:cent_w+half_width_w] = 1
    
    return img * mask 

def get_high_frequency(img: torch.Tensor,
                       factor: Optional[float]=0.1,
                       out_normalize: Optional[bool]=False,
                       mode: Optional[str]='NCHW') -> torch.Tensor:
    
    kspace = fft2d(img, mode=mode)
    h_freq = hp_filter(kspace, factor)
    recon = ifft2d(h_freq, mode=mode).real
    if out_normalize:
        recon = normalize(recon)
    return recon

def get_low_frequency(img: torch.Tensor,
                      factor: Optional[float]=0.9,
                      out_normalize: Optional[bool]=False,
                      mode: Optional[str]='NCHW') -> torch.Tensor:
    
    kpsace = fft2d(img, mode=mode)
    l_freq = lp_filter(kpsace, factor)
    recon = ifft2d(l_freq, mode=mode).real
    if out_normalize:
        recon = normalize(recon)
    return recon
