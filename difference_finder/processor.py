import torch
from difference_finder.utils import get_high_frequency, get_low_frequency

# -------------- Pre-processor ------------- #
PREPROCESS = {}

def register_preprocessor(name: str):
    def wrapper(cls):
        if PREPROCESS.get(name) is not None:
            raise NameError(f"Preprocessor {name} is already registered.")
        PREPROCESS[name] = cls
        return cls
    return wrapper

def get_preprocessor(name: str):
    if PREPROCESS.get(name) is None:
        raise NameError(f"Preprocessor {name} does not exist.")
    return PREPROCESS.get(name)

@register_preprocessor(name='identity')
def identity(img: torch.Tensor) -> torch.Tensor:
    return img

@register_preprocessor(name='normalize')
def normalize(img: torch.Tensor) -> torch.Tensor:
    assert (img.max() > img.min()),\
        print('Maximum of given image is smaller than its minimum.')
    return (img - img.min()) / (img.max() - img.min())

@register_preprocessor(name='highpass_filter')
def high_pass_filer(img: torch.Tensor) -> torch.Tensor:
    # TODO: how to control factor?
    return get_high_frequency(img) 

@register_preprocessor(name='lowpass_filter')
def low_pass_filter(img: torch.Tensor) -> torch.Tensor:
    return get_low_frequency(img)

# -------------- Post-processor ------------- #
POSTPROCESS = {}

def register_postprocessor(name: str):
    def wrapper(cls):
        if POSTPROCESS.get(name) is not None:
            raise NameError(f"Postprocessor {name} is already registered.")
        POSTPROCESS[name] = cls
        return cls
    return wrapper

def get_postprocessor(name: str):
    if POSTPROCESS.get(name) is None:
        raise NameError(f"Postprocessor {name} does not exist.")
    return POSTPROCESS.get(name)

@register_postprocessor(name='identity')
def identity(img: torch.Tensor) -> torch.Tensor:
    return img