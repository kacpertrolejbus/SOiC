import numpy as np
import matplotlib.pyplot as plt

# Funkcje do interpolacji
def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(x - 1)

def f3(x):
    return np.sign(np.sin(8 * x))

# Definicje jąder konwolucji
def kernel_h1(size):
    x = np.linspace(-size // 2, size // 2, size, endpoint=False)
    return np.where((x >= 0) & (x < 1), 1, 0)

def kernel_h2(size):
    x = np.linspace(-size // 2, size // 2, size, endpoint=False)
    return np.where((x >= -0.5) & (x <= 0.5), 1, 0)

def kernel_h3(size):
    x = np.linspace(-size // 2, size // 2, size, endpoint=False)
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)

# Funkcja do obliczania jednowymiarowej konwolucji
def convolve(signal, kernel):
    pad_width = len(kernel) // 2
    padded_signal = np.pad(signal, pad_width, mode='edge')
    convolved = np.convolve(padded_signal, kernel, mode='valid')
    return convolved
    #wykonuje operacje konwolucji sygnału z jądrem
    #dodaje odpowiednie wypełnienie (padding) na krawędziach sygnału, aby wynik miał odpowiedni rozmiar
    #stosuje konwolucję za pomocą np.convolve() z trybem 'valid', by uniknąć problemów z granicami

# Funkcja do obliczenia błędu MSE
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
    #mse(): oblicza błąd średnikwadratowy (MSE) między wartościami prawdziwymi ytrue a przewidzianymi ypred

# Funkcja do interpolacji z konwolucją 
def interpolate_convolution(x, y, factor, kernel):
    #funkcja powyżej używa konwolucji do interpolacji:
    #1. tworzy nowe próbki na osi x o większej gęstości (factor to współczynnik zwiększenia liczby próbek) 
    new_x = np.linspace(x.min(), x.max(), len(x) * factor)
    #2. normalizuje jądro konwolucji (jego suma wynosi 1, co zapobiega skalowaniu sygnału)
    kernel = kernel / kernel.sum()
    #3. oblicza wartości y po konwolucji
    y_interpolated = convolve(y, kernel)
    #4. zwraca nowe próbki x i interpolowane wartości y
    return new_x, np.interp(new_x, x, y_interpolated)
    

# Parametry i konfiguracje
x_range = np.linspace(-np.pi, np.pi, 100)
# wektor próbek x w przedziale [-pi, p] z 100 punktami
kernels = [kernel_h1(5), kernel_h2(5), kernel_h3(5)]
# lista tzrech różnych jąder konwolucji
functions = [f1, f2, f3]
# lista funkcji testowych do interpolacji
factors = [2, 4, 10]
# współczynniki powiększenia liczby próbek (np. podwojenie, czterokrotność, dziesięciokrotność)

# Wykonanie interpolacji i analiza
results = {}

for func_idx, func in enumerate(functions):#iteruje po funkcjach (f1, f2, f3)
    x = x_range
    y = func(x)
    func_name = func.__name__
    results[func_name] = {}

    for kernel_idx, kernel in enumerate(kernels):#dla każdej funkcji wykonuje pętle po jądrach konwolucji(h1, h2, h3)
        kernel_name = f'kernel_h{kernel_idx+1}'
        results[func_name][kernel_name] = {}

        for factor_idx, factor in enumerate(factors):#dla każdego jądra testuje różne współczynniki interpolacji(2,4,10)
            new_x, new_y = interpolate_convolution(x, y, factor, kernel)#wyznacza nowe punkty x
            true_y = func(new_x)#wyznacza nowe interpolowane punkty y
            error = mse(true_y, new_y)#oblicza błąd MSE

            # Zapis wyników
            results[func_name][kernel_name][f'punkty_razy_{factor}'] = error

            # Wykres
            plt.figure(figsize=(8, 6))
            plt.plot(x, y, 'o', label='Punkty oryginalne')
            plt.plot(new_x, true_y, label='Prawdziwa funkcja')
            plt.plot(new_x, new_y, '--', label='Interpolowane')
            plt.title(f'{func_name} | {kernel_name} | Więcej punktów: {factor}')
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(True)
            plt.show()

# Wyświetlenie wyników
for func_name, kernel_results in results.items():
    print(f"Funkcja: {func_name}")
    for kernel_name, factor_results in kernel_results.items():
        print(f"  {kernel_name}:")
        for factor, error in factor_results.items():
            print(f"    {factor}: MSE = {error:.4f}")