import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from PIL import Image

def kernel(points: NDArray, offset: NDArray, width: float) -> NDArray:
    # Obliczam różnicę między punktami a przesunięciem
    roznica = points - offset
    # Sprawdzam, czy różnice w obu wymiarach mieszczą się w zakresie [0, szerokość)
    warunek = (0 <= roznica[:, 0]) & (roznica[:, 0] < width) & (0 <= roznica[:, 1]) & (roznica[:, 1] < width)
     # Zwrócenie tablicy float, gdzie 1.0 oznacza obecność w jądrze
    return warunek.astype(float) 

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
            wartosci = kernel(siatka, offset=punkt, width=w) * value
            # Wartości jądra są dodawane do docelowych pozycji obrazu interpolowanego
            interpolowany_obraz += wartosci.reshape(target_shape)
    return interpolowany_obraz  # Funkcja zwraca interpolowany obraz

if __name__ == "__main__":
    
    image_path = 'file.png'  
     # Otwarcie obrazu i konwersja do skali szarości
    image = Image.open(image_path).convert('L') 
    # Konwersja obrazu do tablicy NumPy
    image_array = np.array(image)  
    # Wykonanie interpolacji z określonym współczynnikiem skalowania
    # Współczynnik skalowania dla interpolacji
    scaling_factor = 2  
    # Interpolacja obrazu
    interpolated_image = image_interpolate2d(image_array, scaling_factor)  

    plt.figure(figsize=(10, 5))  
    plt.subplot(1, 2, 1)  
    plt.title("Oryginalny obraz")  
    plt.imshow(image_array, cmap='gray')  
    plt.axis('off') 
    plt.subplot(1, 2, 2)  
    plt.title("Interpolowany obraz")  
    plt.imshow(interpolated_image, cmap='gray')  
    plt.axis('off')  
    plt.show()  
