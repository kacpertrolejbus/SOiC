import numpy as np
import cv2
import matplotlib.pyplot as plt

path = "namib-noised.png"
image = cv2.imread(path)
imrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def odszumianie_kanal(kanal, radius=100):
    fourier = np.fft.fft2(kanal)
    fourier_s = np.fft.fftshift(fourier)
    rows, cols = kanal.shape
    row, col = rows//2, cols//2
    maska = np.zeros((rows,cols), np.uint8)
    cv2.circle(maska, (col, row), radius, 1, thickness=-1)
    fs = fourier_s*maska
    fourier_s = np.fft.ifftshift(fs)
    kanal_f = np.fft.ifft2(fourier_s)
    return np.abs(kanal_f).clip(0,255).astype(np.uint8)

r, g, b =cv2.split(imrgb)
rf = odszumianie_kanal(r, radius=100)
gf = odszumianie_kanal(g, radius=100)
bf = odszumianie_kanal(b, radius=100)
scalanie = cv2.merge((rf,gf,bf))

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Obraz oryginalny")
plt.imshow(imrgb)

plt.subplot(1,2,2)
plt.title("Obraz po odszumianiu")
plt.imshow(scalanie)

plt.show()

