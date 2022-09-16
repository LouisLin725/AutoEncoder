# AutoEncoder
Using the autoencoder on MNIST dataset.

## Table of contents
* [Image Info](#image-info)
* [Training Models](#training-models)
* [Results](#results)
* [Hardware Equipment](#hardware-equipment)
* [How to run](#how-to-run)
* [Reference](#reference)

## Brief Introduction
In this project, I just tried to familiar with the structure and the implementation of AutoEncoder.
There are two  topics in this work. The first one is about the reconstruction of the MNIST image via autoencoder.
The second one is about using the latent code (vector) to produce the MNIST images through the decoder.   

## Image Info
I used MNIST database as the training and testing images.
- Training images: 60000 
- Testing images: 10000
- Image size: 28 * 28

### MNIST example:
![mnist_images](https://user-images.githubusercontent.com/101628791/190708530-31c45b03-86df-4860-9c02-7b218051ba11.png)

## Training Models
- AutoEncoder
- Encoder
- Decoder


## Results
### Reconstruction
Using the autoencoder to reconstruct the input MNIST images.
![autoencoder](https://user-images.githubusercontent.com/101628791/190708963-e5ba0b98-07a8-407d-8861-370b3847a95d.png)

### Generated images


## Hardware Equipment
* CPU: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz   2.81 GHz
* RAM: 12.0 GB (4+8)
* GPU: NVIDIA GeForce GTX 1050 / Intel(R) HD Graphics 630
* GPU Memory: 2GB GDDR5 (128-bit)

## How to run
