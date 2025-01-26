import io
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Your code here

    # Отцентруем каждую строчку матрицы
    mean_values = np.mean(matrix, axis=1)
    centered_matrix = matrix - mean_values[:, None]
    # Найдем матрицу ковариации
    covariance = np.cov(centered_matrix)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_val, eig_vec = np.linalg.eigh(covariance)
    # Посчитаем количество найденных собственных векторов
    num_eig_vec = eig_vec.shape[1]
    # Сортируем собственные значения в порядке убывания
    sorted_indexes = np.argsort(eig_val)[::-1]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    eig_vec = eig_vec[:, sorted_indexes]
    # Оставляем только p собственных векторов
    eig_vec = eig_vec[:, :p]
    # Проекция данных на новое пространство
    projected = eig_vec.T @ centered_matrix
    return eig_vec, projected, mean_values


def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!

        # Your code here
        eig_vec, projected, mean_values = comp
        decompressed_channel = eig_vec @ projected + mean_values[:, None]
        result_img.append(decompressed_channel)
    return np.clip(np.stack(result_img, axis=2), 0, 255).astype(np.uint8)


def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            # Your code here
            eig_vec, projected, mean_values = pca_compression(img[:, :, j], p)
            compressed.append((eig_vec, projected, mean_values))
        decompressed_img = pca_decompression(compressed)
        axes[i // 3, i % 3].imshow(decompressed_img)
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    # Your code here
    ycbcr_from_rgb = np.array([[0.299, 0.587, 0.114],
                               [-0.1687, -0.3313, 0.5],
                               [0.5, -0.4187, -0.0813]])
    bias = np.array([[0], [128], [128]]).T
    img = np.dot(img, ycbcr_from_rgb.T) + bias
    return np.clip(img, 0, 255).astype(np.uint8)


def ycbcr2rgb(img):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    # Your code here
    rgb_from_ycbcr = np.array([[1, 0, 1.402],
                               [1, -0.34414, -0.71414],
                               [1, 1.772, 0]])
    bias = np.array([[0, -128, -128]])
    img = np.dot(img + bias, rgb_from_ycbcr.T)
    return np.clip(img, 0, 255).astype(np.uint8)


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img[:, :, 1] = gaussian_filter(ycbcr_img[:, :, 1], sigma=10)
    ycbcr_img[:, :, 2] = gaussian_filter(ycbcr_img[:, :, 2], sigma=10)
    new_rgb_img = ycbcr2rgb(ycbcr_img)
    plt.imshow(new_rgb_img)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    ycbcr_img = rgb2ycbcr(rgb_img)
    ycbcr_img[:, :, 0] = gaussian_filter(ycbcr_img[:, :, 0], sigma=10)
    new_rgb_img = ycbcr2rgb(ycbcr_img)
    plt.imshow(new_rgb_img)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """

    # Your code here
    blurred_component = gaussian_filter(component, sigma=10)
    return blurred_component[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    # Your code here
    dct_block = np.zeros((8, 8))
    for u in range(8):
        for v in range(8):
            alpha_u = np.sqrt(0.5) if u == 0 else 1
            alpha_v = np.sqrt(0.5) if v == 0 else 1
            sum = 0
            for x in range(8):
                for y in range(8):
                    sum += block[x, y] * np.cos((2*x + 1) * u * np.pi / 16) * np.cos((2*y + 1) * v * np.pi / 16)
            dct_block[u, v] = 0.25 * alpha_u * alpha_v * sum
    return dct_block


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    # Your code here

    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    # Your code here
    if 1 <= q < 50:
        S = 5000 / q
    elif 50 <= q <= 99:
        S = 200 - 2 * q
    else:
        S = 1
    quant = np.floor((50 + S * default_quantization_matrix) / 100)
    quant[quant == 0] = 1
    return quant


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    # Your code here
    res = [[] for _ in range(15)]
    for i in range(8):
        for j in range(8):
            sum = i + j
            if (sum % 2 == 0):
                res[sum].insert(0, block[i][j])
            else:
                res[sum].append(block[i][j])
    zigzag_list = []
    for i in res:
        for j in i:
            zigzag_list.append(j)
    return zigzag_list


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    # Your code here
    compressed = []
    zero_count = 0

    for i in zigzag_list:
        if i == 0:
            zero_count += 1
        else:
            if zero_count > 0:
                compressed.append(0)
                compressed.append(zero_count)
                zero_count = 0
            compressed.append(i)
    
    if zero_count > 0:
        compressed.append(0)
        compressed.append(zero_count)
    return compressed


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """
    img = rgb2ycbcr(img).astype(np.float64)

    compressed = []
    for k in range(3):
        temp = 1 if k == 0 else 2
        n_rows = img.shape[0] // temp
        n_cols = img.shape[1] // temp
        compressed.append([])
        for i in range(0, n_rows, 8):
            for j in range(0, n_cols, 8):
                block = img[i * temp:(i + 8) * temp, j * temp:(j + 8) * temp, k] - 128
                dct_block = dct(block)
                quant = quantization(dct_block, quantization_matrixes[1 if k > 0 else 0])
                zigzag_list = zigzag(quant)
                compressed_list = compression(zigzag_list)
                compressed[k].append(compressed_list)
    return compressed


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    # Your code here
    i = 0
    decompressed_list = []
    while i < len(compressed_list):
        if compressed_list[i] == 0:
            decompressed_list.extend([0] * compressed_list[i + 1])
            i += 2
        else:
            decompressed_list.append(compressed_list[i])
            i += 1
    return decompressed_list


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    # Your code here
    block = np.zeros((8, 8))
    ind = 0
    for sum in range(15):
        if sum % 2 == 0:
            i = min(sum, 7)
            j = sum - i
            while i >= 0 and j < 8:
                block[i, j] = input[ind]
                ind += 1
                i -= 1
                j += 1
        else:
            i = max(0, sum - 7)
            j = sum - i
            while j >= 0 and i < 8:
                block[i, j] = input[ind]
                ind += 1
                i += 1
                j -= 1
    return block


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    # Your code here

    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    # Your code here
    old = np.zeros((8, 8))
    for x in range(8):
        for y in range(8):
            sum = 0
            for u in range(8):
                for v in range(8):
                    alpha_u = np.sqrt(0.5) if u == 0 else 1
                    alpha_v = np.sqrt(0.5) if v == 0 else 1
                    sum += alpha_u * alpha_v * block[u, v] * np.cos((2*x + 1) * u * np.pi / 16) * np.cos((2*y + 1) * v * np.pi / 16)
            old[x, y] = 0.25 * sum
    return np.round(old)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    # Your code here
    return np.repeat(np.repeat(component, 2, axis=1), 2, axis=0)

def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    # Your code here
    ycbcr_components = []
    for k in range(3):
        temp = 1 if k == 0 else 2
        n_rows, n_cols = result_shape[0] // temp, result_shape[1] // temp
        comp = np.zeros((n_rows, n_cols))
        ind = 0
        for i in range(0, n_rows, 8):
            for j in range(0, n_cols, 8):
                compressed_list = result[k][ind]
                ind += 1
                decompressed = inverse_compression(compressed_list)
                block = inverse_zigzag(decompressed)
                dequant = inverse_quantization(block, quantization_matrixes[1 if k == 0 else 0])
                comp[i : i + 8, j : j + 8] = inverse_dct(dequant) + 128
        ycbcr_components.append(comp)
    ycbcr_components[1] = upsampling(ycbcr_components[1][:, :, None])[:, :, 0]
    ycbcr_components[2] = upsampling(ycbcr_components[2][:, :, None])[:, :, 0]
    ycbcr_image = np.dstack((ycbcr_components[0], ycbcr_components[1], ycbcr_components[2]))
    return ycbcr2rgb(ycbcr_image.astype(np.uint8))


def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        y_quant_matrix = own_quantization_matrix(y_quantization_matrix, p)
        color_quant_matrix = own_quantization_matrix(color_quantization_matrix, p)
        quantization_matrices = [y_quant_matrix, color_quant_matrix]
        compressed_img = jpeg_compression(img, quantization_matrices)
        decompressed_img = jpeg_decompression(compressed_img, img.shape, quantization_matrices)

        axes[i // 3, i % 3].imshow(decompressed_img)
        axes[i // 3, i % 3].set_title("Quality Factor: {}".format(p))

    fig.savefig("jpeg_visualization.png")


def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes

    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
