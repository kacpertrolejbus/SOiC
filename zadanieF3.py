import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

path = "file.png"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255
image+=np.random.randn(image.shape[0] * image.shape[1]).reshape(image.shape)

falka = pywt.dwt2(image, 'haar')
cA, (cH, cV, cD) = falka
treshold = 100
cH = pywt.threshold(cH, treshold, mode='soft')
cV = pywt.threshold(cV, treshold, mode='soft')
cD = pywt.threshold(cD, treshold, mode='soft')

odszumianie = (cA, (cH,cV,cD))
imgo = pywt.idwt2(odszumianie, 'haar')
imgo = np.clip(imgo,0,255).astype(np.uint8)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Obraz oryginalny")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Obraz po odszumianiu")
plt.imshow(imgo, cmap='gray')
plt.axis('off')

plt.show()
