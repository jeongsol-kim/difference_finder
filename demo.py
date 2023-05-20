
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from difference_finder.utils import to_np
from difference_finder.finder import Finder

def main():
    img1 = Image.open('samples/imgdir1/img2_1.png').convert('RGB')
    img2 = Image.open('samples/imgdir2/img2_2.png').convert('RGB')

    img1 = ToTensor()(img1).unsqueeze(0)[...,1:, 1:]
    img2 = ToTensor()(img2).unsqueeze(0)[...,:-1,:-1]

    worker = Finder(pre_processor='identity', strategy='gradient', metric='psnr')
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
    plt.savefig('test.png')



if __name__ == '__main__':
    main()
