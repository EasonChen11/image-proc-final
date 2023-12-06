import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img

# Load the image
image_path = "refrigerator_gray.png"  # Replace with the actual image path
image = Image.open(image_path).convert('L')  # Open the image and convert to grayscale

# Perform FFT
fft = np.fft.fft2(image)
fft_shift = np.fft.fftshift(fft)

# Display the FFT
plt.subplot(1, 2, 1)
plt.imshow(np.log(1 + np.abs(fft)), cmap='gray')
plt.title('FFT')
plt.axis('off')

# Display the FFTshift
plt.subplot(1, 2, 2)
plt.imshow(np.log(1 + np.abs(fft_shift)), cmap='gray')
plt.title('FFTshift')
plt.axis('off')

plt.show()
