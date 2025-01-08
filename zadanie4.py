import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from PIL import Image

def keys_kernel(points, offset, width):
    # Obliczam róznice współrzędnych x i y
    x_roznica = np.abs(points[:, 0] - offset[0])
    y_roznica = np.abs(points[:, 1] - offset[1])
    # Obliczam wagi oparte na odległościach punktu od przesunięcia w zakresie jądra
    waga = (1 - x_roznica / width) * (1 - y_roznica / width)
    # Zwracam tylko wagi które mieszczą się w zakresie jądra, reszta a ustawione 0
    return np.where((x_roznica < width) & (y_roznica < width), waga, 0)

def image_interpolate2d(image: NDArray, ratio: int) -> NDArray:
     # Pobieram wymiary oryginalnego obrazu 
    height, width = image.shape 
    # Obliczam wymiary inerpolowanego obrazu
    target_shape = (height * ratio, width * ratio)  
    # Dla interpolowanego obrazu tworzę siatkę punktów
    siatka = np.array([
        [i / ratio, j / ratio]  # Każdy punkt w przesstrzeni interpolacji jest mapowanym do jego żrófłowych wartości
        for i in range(target_shape[0])  # Iteruje po wszystkich wierszach docelowego obrazu
        for j in range(target_shape[1])  # Iteruje po wszytskich kolumnach obrazu docelowego
    ])
    # Inicjalizacja pustej macierzy dla interpolowanego obrazu
    interpolowany_obraz = np.zeros(target_shape)
    # Ustawiam szerokość jądra
    w = 1
    # Proces interpolacji
     # Iteruje po kązdym pikselu oryginalnego obrazu
    for i in range(height): 
        for j in range(width):
            punkt = np.array([i, j])  # Pozycja aktualnego piksela
            value = image[i, j] 
            # Obliczenie wartości jądra dla wszystkich punktów siatki interpolacji i skalowanie ich przez wartość piksela
            wartosci = keys_kernel(siatka, offset=punkt, width=w) * value
            # Wartości jądra są dodawane do docelowych pozycji obrazu interpolowanego
            interpolowany_obraz += wartosci.reshape(target_shape)
    return interpolowany_obraz  # Funkcja zwraca interpolowany obraz

if __name__ == "__main__":
    image_path = "file.png"
    # Konwersja obrazu do skali szarości
    image = Image.open(image_path).convert('L')
    # Konwersja obrazu do tablicy NumPy
    image_array = np.array(image) 
    # Interpolacja obrazu z określonym współczynnikiem skalowania
    # Współczynnik skalowania
    scaling_factor = 2 
    interpolated_image = image_interpolate2d(image_array, scaling_factor)

    plt.figure(figsize=(10, 5)) 
    plt.subplot(1, 2, 1)
    plt.title("Oryginalny obraz")
    plt.imshow(image_array, cmap='gray')  
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Interpolowany obraz (Keys Kernel)")
    plt.imshow(interpolated_image, cmap='gray') 
    plt.axis('off')
    plt.show()
