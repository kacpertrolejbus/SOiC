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

def interpolate_color_image(image: NDArray, ratio: int) -> NDArray:
    # Tworzę tabelę dla interpolowanego obrazu kolorowego
    kanaly = image.shape[2] 
    itp_kanaly = []
    # Iteruje przez każdy kanał i interpolacja
    for c in range(kanaly):
        # Pobranie pojedynczego kanału
        ch = image[:, :, c]  
        # Interpolacja pojedynczego kanału
        itp_ch = image_interpolate2d(ch, ratio)  
        # Dodanie zinterpolowanego kanału do listy
        itp_kanaly.append(itp_ch)  
    # Połączenie zinterpolowanych kanałów w obraz RGB
    interpolowany_obraz = np.stack(itp_kanaly, axis=-1)
    return interpolowany_obraz

if __name__ == "__main__":
    image_path = 'panda.jpg'  
    image = Image.open(image_path)  
    img_a = np.array(image)  # Konwersja obrazu do tablicy NumPy
    # Sprawdzam, czy obraz jest kolorowy czy w skali szarości
    if len(img_a.shape) == 2:  # Obraz w skali szarości
        skalowanie = 2  # Współczynnik skalowania dla interpolacji
        interpolated_image = image_interpolate2d(img_a, skalowanie)  # Interpolacja obrazu w skali szarości
    else:  # Obraz kolorowy (RGB)
        skalowanie = 2  # Współczynnik skalowania dla interpolacji
        itp_img = interpolate_color_image(img_a, skalowanie)  # Interpolacja obrazu kolorowego

    
    plt.figure(figsize=(10, 5))  
    plt.subplot(1, 2, 1)  
    plt.title("Oryginalny obraz")  
    plt.imshow(img_a, cmap='gray' if len(img_a.shape) == 2 else None)  
    plt.axis('off')  
    plt.subplot(1, 2, 2)  
    plt.title("Interpolowany obraz")  
    plt.imshow(itp_img.astype(np.uint8), cmap='gray' if len(img_a.shape) == 2 else None)  
    plt.axis('off')  
    plt.show() 
