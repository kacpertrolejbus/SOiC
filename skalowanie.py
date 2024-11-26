import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Funkcje skalowania obrazu

def downscale_image(image, kernel):
    """Zmniejszanie obrazu za pomocą splotu z jądrem uśredniającym."""
    kernel = kernel / kernel.sum()  # Normalizacja jądra, macierz opisująca sposób obliczania nowej wartości piksela (tu: jądro uśredniające), normalizacja sprawia że suma wag wynosi 1
    h, w = image.shape #Rozmiary obrazu
    kh, kw = kernel.shape #Rozmiary jądra 
    pad_h, pad_w = kh // 2, kw // 2 #Obliczanie ilości paddingu

    # Padding obrazu - ) dodaje piksele na brzegach obrazu. Tryb reflect powiela wartości w sposób lustrzany, aby uniknąć problemów na krawędziach
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # Splot - każdy piksel jest uśredniany w oparciu o sąsiedztwo określone przez jądro (kernel)
    convolved = np.zeros_like(image) # tworzymy pusty obraz wynikowy
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + kh, j:j + kw] # sąsiedztwo wokół piksela (i, j)
            convolved[i, j] = np.sum(region * kernel) # mnożenie sąsiedztwa przez jądro
    
    # każdy piksel w nowym obrazie obliczany jest jako suma iloczynu jądra i fragmentu obrazu, nad którym jądro jest umieszczone.

    # Subsampling - pobierane są co drugi wiersz i co druga kolumna obrazu
    return convolved[::2, ::2] # wybieramy co drugi piksel w obu wymiarach

    #efektem jest obraz o rozdzielczości 2-krotnie mniejszej (wysokość i szerokość).

def upscale_image(image, method='linear'): # powiększanie odbywa się za pomocą interpolacji dwuliniowej. Każdy nowy piksel jest obliczany jako średnia ważona wartości z najbliższych pikseli w oryginalnym obrazie.
    """Powiększanie obrazu za pomocą interpolacji funkcji."""
    scale_factor = 2 # skalowanie o czynnik 2x
    new_shape = (image.shape[0] * scale_factor, image.shape[1] * scale_factor)
    zoomed_image = np.zeros(new_shape) # tworzenie pustego obrazu wynikowego

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            orig_x, orig_y = i / scale_factor, j / scale_factor # Miejsce w oryginalnym obrazie
            x0, y0 = int(orig_x), int(orig_y) # Najbliższy górny lewy piksel
            x1, y1 = min(x0 + 1, image.shape[0] - 1), min(y0 + 1, image.shape[1] - 1) # Dolny prawy 
            #dla każdego nowego piksela (i, j) obliczane są jego współrzędne w oryginalnym obrazie (orig_x, orig_y) 

            dx, dy = orig_x - x0, orig_y - y0 # Odległości od pikseli bazowych
            zoomed_image[i, j] = (
                (1 - dx) * (1 - dy) * image[x0, y0] +
                dx * (1 - dy) * image[x1, y0] +
                (1 - dx) * dy * image[x0, y1] +
                dx * dy * image[x1, y1]
            ) # nowa wartość to średnia ważona na podstawie odległości (dx, dy) od najbliższych pikseli.

    return zoomed_image

# średni błąd kwadratowy (MSE) mierzy różnicę między dwoma obrazam
def mse(image1, image2):
    """Obliczanie błędu średniokwadratowego (MSE) między dwoma obrazami."""
    return np.mean((image1 - image2) ** 2)

# Główna część programu
if __name__ == "__main__":
    # Wczytanie obrazu w skali szarości
    image_path = '60836_escape_from_the_shadows.jpg'  # Ścieżka do obrazu drzewa
    original_image = Image.open(image_path).convert('L')  # Konwersja do skali szarości
    original_image = np.array(original_image, dtype=np.float32) / 255.0  # Normalizacja do [0, 1]

    # Jądro uśredniające
    kernel_avg = np.ones((3, 3)) / 9

    # Zmniejszanie obrazu
    reduced_image = downscale_image(original_image, kernel_avg)

    # Powiększanie obrazu
    upscaled_image = upscale_image(reduced_image)

    # Obliczanie MSE
    restored_shape = original_image.shape
    mse_value = mse(original_image[:restored_shape[0], :restored_shape[1]], upscaled_image[:restored_shape[0], :restored_shape[1]])

    # Wizualizacja wyników
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Oryginalny obraz")

    axes[1].imshow(reduced_image, cmap="gray")
    axes[1].set_title("Zmniejszony obraz")

    axes[2].imshow(upscaled_image, cmap="gray")
    axes[2].set_title(f"Powiększony obraz\nMSE = {mse_value:.4f}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Wynik MSE
    print(f"MSE: {mse_value:.4f}")
