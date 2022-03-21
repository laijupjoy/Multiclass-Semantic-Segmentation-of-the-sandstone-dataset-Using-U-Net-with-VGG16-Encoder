# Multiclass-Semantic-segmentation-of-the-sandstone-dataset

This repository consists of files required for Multiclass Semantic Segmentation of the sandstone dataset.

## Project Tasks:

1. Convert tiff image and mask stack into small patches of size 128*128
2. Create multi class U-Net model architecture.
3. Train the U-Net model with 128*128 size images and masks.
4. Generate the model with minimum loss.
5. Find the mean IoU
6. Find the IoU values for each classes.
7. Predict segmentation of large images.

## Pixels Categories in an Image:

<p align="center">
  <img src="images\pixel classes.png" alt="workflow"/>
</p>

## Histogram of Pixels Categories in an Image:

<p align="center">
  <img src="images\histogram of large image labels.png" alt="workflow"/>
</p>


## IoU:

<p align="center">
  <img src="images\iou.png" alt="workflow"/>
</p>

## Loss:

<p align="center">
  <img src="images\loss.png" alt="workflow"/>
</p>

## :Result after 25 Epochs

<p align="center">
  <img src="images\summary.png" alt="workflow"/>
</p>


## Segmentation Results:

<p align="center">
  <img src="images\128.png" alt="workflow"/>
</p>

<p align="center">
  <img src="images\large.png" alt="workflow"/>
</p>
