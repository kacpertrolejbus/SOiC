import numpy as np  
import matplotlib.pyplot as plt  
from scipy.ndimage import convolve  
from skimage import io, transform, color  

# Wczytanie obrazu i przekształcenie do skali szarości dla konwolucji
image = io.imread('jez.jpg')[:, :, :3]  # Załadowanie obrazu i usunięcie kanału alfa
image = transform.resize(image, (1024, 1024, 3))  # Skalowanie obrazu
image_gray = color.rgb2gray(image)  # Konwersja do skali szarości

# Definicja filtrów konwolucyjnych
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # Operator Sobela w kierunku X
blur = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])  # Rozmycie Gaussowskie
sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Filtr wyostrzający

# Funkcja do zastosowania filtru konwolucyjnego
def apply_filter(image, kernel, title):
    filtered = convolve(image, kernel)  # Konwolucja na skali szarości
    plt.figure(figsize=(6, 6))
    plt.imshow(filtered, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Wykrywanie krawędzi, rozmywanie i wyostrzanie obrazu
apply_filter(image_gray, sobel_x, 'Wykrywanie krawędzi - Sobel X')
apply_filter(image_gray, blur, 'Rozmywanie obrazu')
apply_filter(image_gray, sharpen, 'Wyostrzanie obrazu')

# Definicja maski Bayera
bayer_mask = np.array([[[0, 1], [0, 0]],  # Czerwony
                        [[1, 0], [0, 1]],  # Zielony
                        [[0, 0], [1, 0]]], dtype=np.uint8)  # Niebieski
bayer_mask = np.transpose(bayer_mask, axes=(1, 2, 0))

# Funkcja do tworzenia CFA dla obrazu
def color_filter_array(mask, shape):
    return np.dstack([
        np.tile(mask[:, :, channel], np.asarray(shape) // len(mask[:, :, channel]))
        for channel in range(mask.shape[-1])
    ])

# Tworzenie filtru Bayera
bayer_filter = color_filter_array(bayer_mask, np.array([1024, 1024]))

# Symulacja obrazu z czujnika
sensor_image = image * bayer_filter

# Filtr konwolucyjny do demozaikowania
demosaicking_convolution_mask = np.dstack([
    np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]]),  # R
    np.array([[0.5, 1, 0.5], [1, 4, 1], [0.5, 1, 0.5]]) / 4,  # G
    np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]])   # B
])

# Implementacja demozaikowania
reconstructed_image = np.dstack([
    convolve(sensor_image[:, :, channel], demosaicking_convolution_mask[:, :, channel], mode="nearest")
    for channel in range(3)
])

# Normalizacja wartości do przedziału [0, 1]
reconstructed_image = np.clip(reconstructed_image, 0, 1)

# Wizualizacja wyniku demozaikowania
plt.figure(figsize=(6, 6))
plt.imshow(reconstructed_image)  # Wyświetlenie w kolorze
plt.title('Demozaikowanie - Filtr Bayera')
plt.axis('off')
plt.show()
