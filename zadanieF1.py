import numpy as np
import cv2
import matplotlib.pyplot as plt

path = "circle-noised.png"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

Fourier = np.fft.fft2(image)
Fourier_s = np.fft.fftshift(Fourier)

widmo = 20 * np.log(np.abs(Fourier_s) + 1)

rows, cols = image.shape
row, col = rows // 2, cols // 2
radius = 100

maska = np.zeros((rows, cols), np.uint8)
cv2.circle(maska, (col, row), radius, 1, thickness=-1)
filtr = Fourier_s * maska

Fourier_s = np.fft.ifftshift(filtr)
image_f = np.fft.ifft2(Fourier_s)
image_f = np.abs(image_f)

plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.title("Obraz oryginalny")
plt.imshow(image, cmap='gray')

plt.subplot(1,3,2)
plt.title("Widmo amplitudowe")
plt.imshow(widmo, cmap='gray')

plt.subplot(1,3,3)
plt.title("Obraz po odszumianiu")
plt.imshow(image_f, cmap='gray')

plt.show()