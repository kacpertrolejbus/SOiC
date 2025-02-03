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
bayer_mask = np.zeros((2, 2, 3))
bayer_mask[0, 0, 0] = 1  # Czerwony
bayer_mask[0, 1, 1] = 1  # Zielony
bayer_mask[1, 0, 1] = 1  # Zielony
bayer_mask[1, 1, 2] = 1  # Niebieski

# Powielenie maski Bayera na cały obraz
bayer_filter = np.tile(bayer_mask, (image.shape[0] // 2, image.shape[1] // 2, 1))

# Symulacja obrazu z czujnika
sensor_image = image * bayer_filter

# Filtry interpolacyjne dla demozaikowania
interp_kernels = {
    'red': np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]]),
    'green': np.array([[0, 0.25, 0], [0.25, 1, 0.25], [0, 0.25, 0]]),
    'blue': np.array([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]])
}

# Implementacja demozaikowania
reconstructed_image = np.dstack([
    convolve(sensor_image[:, :, 0], interp_kernels['red'], mode="nearest"),
    convolve(sensor_image[:, :, 1], interp_kernels['green'], mode="nearest"),
    convolve(sensor_image[:, :, 2], interp_kernels['blue'], mode="nearest")
])

# Normalizacja wartości do przedziału [0, 1]
reconstructed_image = np.clip(reconstructed_image, 0, 1)

# Wizualizacja wyniku demozaikowania
plt.figure(figsize=(6, 6))
plt.imshow(reconstructed_image)
plt.title('Demozaikowanie - Filtr Bayera')
plt.axis('off')
plt.show()