import math
from PIL import Image
import optparse
import numpy as np
from numpy import fft as fft
import matplotlib.pyplot as plt
import matplotlib.image as img
# Read the image
image_path = "refrigerator.png"  # Replace with the actual image path
image = Image.open(image_path)

# Convert RGB to grayscale
gray_image = image.convert('L')
# Save the grayscale image
image_path = image_path[:-4] + "_gray.png"
print(image_path)
# save image as image_path
gray_image.save(image_path)