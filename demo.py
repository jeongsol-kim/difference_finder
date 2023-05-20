from pathlib import Path
import matplotlib.pyplot as plt
from difference_finder.finder import Finder

def main():
    first_dir = Path('samples/imgdir1')
    second_dir = Path('samples/imgdir2')

    # define finder
    worker = Finder(pre_processor='identity', strategy='gradient', metric='psnr')
    output = worker.run(first_dir, second_dir)
    
    for i, out in enumerate(output):
        plt.imsave(f'test_{i}.png', out, cmap='inferno')

if __name__ == '__main__':
    main()
