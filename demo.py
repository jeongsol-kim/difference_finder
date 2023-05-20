from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from difference_finder.utils import to_np
from difference_finder.finder import Finder
from difference_finder.data import get_loader

def main():
    first_dir = Path('samples/imgdir1')
    second_dir = Path('samples/imgdir2')

    # load all file paths
    first_files = sorted(first_dir.glob('*.png'))
    second_files = sorted(second_dir.glob('*.png'))
    
    # prepare dataloader
    loader = get_loader(files1=first_files,
                        files2=second_files,
                        transform=ToTensor(),
                        num_workers=0)

    # define finder
    worker = Finder(pre_processor='identity', strategy='gradient', metric='ssim')

    for i, (img1, img2) in enumerate(loader):
        output = worker.run(img1, img2)

        plt.close()
        plt.figure(figsize=(20,12))
        plt.subplot(221)
        plt.imshow(to_np(img1)[0])
        plt.title('Image 1')
        plt.subplot(222)
        plt.imshow(to_np(img2)[0])
        plt.title('Image 2')
        plt.subplot(223)
        plt.imshow(to_np(output)[0].mean(axis=-1), cmap='inferno')
        plt.title('Difference')
        plt.subplot(224)
        plt.imshow(to_np(img1)[0], alpha=0.5)
        plt.imshow(to_np(output)[0].mean(axis=-1), cmap='inferno', alpha=0.5)
        plt.title('Overlay')
        plt.savefig(f'test_{i}.png')



if __name__ == '__main__':
    main()
