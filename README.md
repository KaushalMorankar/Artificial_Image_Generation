# Artificial_Image_Generation

- This project is to generate artificial image of numbers(0-9) from the MNIST dataset.
- Just run the code to generate the artificially generated number
- One can change the number of epochs to get the result quicker but inaccurate.
- noise = torch.randn(1, 100, device='cuda')  Random generation of the digit
- The code is to generate a random number which is there in dataset. If one has to generate a specific number then, one needs to pass the label too as a parameter to both generator and discriminator which will produce the desired number .This GAN is called cGAN (Conditional GAN).  

### Required Libraries
- pip install torch torchvision matplotlib numpy
 
### Try it
- Load other datasets and get the image you want
