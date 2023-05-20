# Difference Finder

This repository is for finding difference between two images automatically.

## Motivation
With the extensive exploration of machine learning techniques, the quality of reconstructed images has reached a saturation point. Consequently, comparing baseline methods has become challenging. While quantitative evaluations may indicate marginal improvements, demonstrating qualitative enhancements has been difficult (in my experience).

<br />


## What you can do
Given two images, this repo provides the difference map. See the below examples.

<br />

### Example 1. Hard "spot the difference" (ref: https://thetem.co.kr/article/%EC%8A%A4%ED%86%A0%EB%A6%AC/2/1801/)

There are *seven* different spots. 
Try by yourself before seeing the result!

Image 1 | Image 2
:-------------------------:|:-------------------------:
![example1_1](samples/imgdir1/img1.png)  | ![example1_2](samples/imgdir2/img2.png)


### Result.

Strategy: Difference | Strategy: Gradient /  Metric: PSNR
:------------------:|:---------------------------------:
![result1_1](figures/test_0_diff.png) | ![result1_2](figures/test_0_grad.png)


<br />

### Example 2. Complex "spot the difference" (ref: https://www.pinterest.co.kr/pin/312578030384336039/)

There are *seven* different spots. 
Try by yourself before seeing the result!

Image 1 | Image 2
:-------------------------:|:-------------------------:
![example1_1](samples/imgdir1/img2_1.png)  | ![example1_2](samples/imgdir2/img2_2.png)


### Result.

Strategy: Difference | Preprocess: HP filter / Strategy: Gradient /  Metric: SSIM
:------------------:|:---------------------------------:
![result1_1](figures/test_1_diff.png) | ![result1_2](figures/test_1_grad.png)