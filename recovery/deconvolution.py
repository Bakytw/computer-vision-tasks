import numpy as np
from scipy.fft import fft2, ifftshift, ifft2


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    center = (size - 1) / 2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - center)**2 + (j - center)**2)
            kernel[i, j] = np.exp(-(r**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    i = shape[0] - h.shape[0]
    j = shape[1] - h.shape[1]
    padding = ((i // 2 + i % 2, i // 2), (j // 2 + j % 2, j // 2))
    h_padded = np.pad(h, padding)
    return fft2(ifftshift(h_padded))


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.zeros_like(H)
    H_inv[np.abs(H) > threshold] = 1 / H[np.abs(H) > threshold]
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G = fft2(blurred_img)
    H_inv = inverse_kernel(fourier_transform(h, blurred_img.shape), threshold)
    F = G * H_inv
    return np.abs(ifft2(F))


def wiener_filtering(blurred_img, h, K=0.00003):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G = fft2(blurred_img)
    H = fourier_transform(h, blurred_img.shape)
    Wiener = np.conjugate(H) / (np.abs(H) ** 2 + K)  # Фильтр Винера
    F = Wiener * G
    return np.abs(ifft2(F))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    mse = np.mean((img1 - img2) ** 2)
    max_i = 255.0
    return 20 * np.log10(max_i / np.sqrt(mse))
