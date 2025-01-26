import numpy as np
from scipy.ndimage import convolve
from numpy.lib.stride_tricks import sliding_window_view

def get_bayer_masks(n_rows, n_cols):
    """
    :param n_rows: `int`, number of rows
    :param n_cols: `int`, number of columns

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.bool_`
        containing red, green and blue Bayer masks
    """
    red = np.array([[False, True], [False, False]])
    green = np.array([[True, False], [False, True]])
    blue = np.array([[False, False], [True, False]])
    r = np.tile(red, (n_rows // 2 + 1, n_cols // 2 + 1))[:n_rows, :n_cols]
    g = np.tile(green, (n_rows // 2 + 1, n_cols // 2 + 1))[:n_rows, :n_cols]
    b = np.tile(blue, (n_rows // 2 + 1, n_cols // 2 + 1))[:n_rows, :n_cols]
    res = np.dstack((r, g, b))
    return res


def get_colored_img(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        each channel contains known color values or zeros
        depending on Bayer masks
    """
    n_rows, n_cols = raw_img.shape
    masks = get_bayer_masks(n_rows, n_cols)
    r = masks[..., 0]
    g = masks[..., 1]
    b = masks[..., 2]
    res = np.zeros((n_rows, n_cols, 3), dtype="uint8")
    res[..., 0] = np.where(r, raw_img, 0)
    res[..., 1] = np.where(g, raw_img, 0)
    res[..., 2] = np.where(b, raw_img, 0)
    return res


def get_raw_img(colored_img):
    """
    :param colored_img:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        colored image

    :return:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image as captured by camera
    """
    r = colored_img[..., 0]
    g = colored_img[..., 1]
    b = colored_img[..., 2]
    n_rows, n_cols, _ = colored_img.shape
    masks = get_bayer_masks(n_rows, n_cols)
    res = np.zeros((n_rows, n_cols), dtype="uint8")
    res[masks[..., 0]] = r[masks[..., 0]]
    res[masks[..., 1]] = g[masks[..., 1]]
    res[masks[..., 2]] = b[masks[..., 2]]
    return res


def bilinear_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
        result of bilinear interpolation
    """

    
    n_rows, n_cols = raw_img.shape
    colored_img = get_colored_img(raw_img)
    res = np.tile(raw_img, (3, 1, 1))
    res = np.dstack(res)
    bayer_masks = get_bayer_masks(n_rows, n_cols)
    for c in range(3):
        colored = sliding_window_view(colored_img[:, :, c], (3, 3)).reshape(-1, 3, 3)
        masked = sliding_window_view(bayer_masks[:, :, c], (3, 3)).reshape(-1, 3, 3)
        non_zero_sum = np.sum(colored * (colored != 0), axis=(1, 2))
        non_zero_count = np.sum(masked, axis=(1, 2))        
        res[1:-1, 1:-1, c] = np.where(non_zero_count != 0, non_zero_sum // non_zero_count, 0).reshape(n_rows - 2, -1)
    res[bayer_masks] = colored_img[bayer_masks]
    return res

def improved_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`, raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        result of improved interpolation
    """
    raw_img = raw_img.astype(np.float64)
    n_rows, n_cols = raw_img.shape
    g_r = g_b = np.array([[0,  0, -1,  0,  0],
                       [0,  0,  2,  0,  0],
                       [-1, 2,  4,  2, -1],
                       [0,  0,  2,  0,  0],
                       [0,  0, -1,  0,  0]])
    r_g_r_row = b_g_b_row = np.array([[0,  0, 1/2,  0,  0],
                                [0, -1, 0, -1, 0],
                                [-1, 4, 5, 4, -1],
                                [0, -1, 0, -1, 0],
                                [0,  0, 1/2,  0,  0]])
    r_g_b_row = b_g_r_row = np.array([[0,  0, -1,  0,  0],
                                [0, -1, 4, -1, 0],
                                [1/2, 0, 5, 0, 1/2],
                                [0, -1, 4, -1, 0],
                                [0,  0, -1,  0,  0]])
    r_b = b_r = np.array([[0, 0, -3/2, 0, 0],
                       [0,    2, 0, 2,  0],
                       [-3/2, 0, 6, 0, -3/2],
                       [0,    2, 0, 2,  0],
                       [0, 0, -3/2, 0, 0]])

    res = np.zeros((n_rows, n_cols, 3), dtype=np.float64)
    masks = get_bayer_masks(n_rows, n_cols)
    r = masks[:, :, 0]
    g = masks[:, :, 1]
    b = masks[:, :, 2]

    def apply_convolution(raw_img, mask, kernel):
        return np.where(mask, convolve(raw_img, kernel / 8.0), 0)

    red_channel = np.where(r, raw_img, 0)
    convolution_g_even = apply_convolution(raw_img, g & (np.arange(n_rows) % 2 == 0)[:, None], r_g_r_row)
    convolution_g_odd = apply_convolution(raw_img, g & (np.arange(n_rows) % 2 == 1)[:, None], r_g_b_row)
    convolution_b = apply_convolution(raw_img, b, r_b)
    res[:, :, 0] = red_channel + convolution_g_even + convolution_g_odd + convolution_b

    green_channel = np.where(g, raw_img, 0)
    convolution_r = apply_convolution(raw_img, r, g_r)
    convolution_b = apply_convolution(raw_img, b, g_b)
    res[:, :, 1] = green_channel + convolution_r + convolution_b

    blue_channel = np.where(b, raw_img, 0)
    convolution_g_even = apply_convolution(raw_img, g & (np.arange(n_rows) % 2 == 0)[:, None], b_g_r_row)
    convolution_g_odd = apply_convolution(raw_img, g & (np.arange(n_rows) % 2 == 1)[:, None], b_g_b_row)
    convolution_r = apply_convolution(raw_img, r, b_r)

    res[:, :, 2] = blue_channel + convolution_g_even + convolution_g_odd + convolution_r
    return np.clip(res, 0, 255).astype(np.uint8)


def compute_psnr(img_pred, img_gt):
    """
    :param img_pred:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param img_gt:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric
    """
    n_rows, n_cols, _ = img_pred.shape
    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)
    mse = np.sum((img_pred - img_gt) ** 2) / (n_rows * n_cols * 3)
    if mse == 0:
        raise ValueError
    return 10 * np.log10(np.max(img_gt** 2)  / mse)

if __name__ == "__main__":
    from PIL import Image

    raw_img_path = "tests/04_unittest_bilinear_img_input/02.png"
    raw_img = np.array(Image.open(raw_img_path))

    img_bilinear = bilinear_interpolation(raw_img)
    Image.fromarray(img_bilinear).save("bilinear.png")

    img_improved = improved_interpolation(raw_img)
    Image.fromarray(img_improved).save("improved.png")
