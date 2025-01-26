import numpy as np

# Read the implementation of the align_image function in pipeline.py
# to see, how these functions will be used for image alignment.


def extract_channel_plates(raw_img, crop):
    a, n_cols = raw_img.shape
    n = a // 3
    unaligned_rgb = [raw_img[2 * n: 3 * n, :], raw_img[n : 2 * n, :], raw_img[:n, :]]
    coords = [np.array([2 * n, 0]), np.array([n, 0]), np.array([0, 0])]
    if not crop:
        return unaligned_rgb, coords

    crop_raws = int(0.1 * n)
    crop_cols = int(0.1 * n_cols)
    for color in range(3):
        raw_start = coords[color][0]
        unaligned_rgb[color] = unaligned_rgb[color][crop_raws : int(0.9 * n), crop_cols : int(0.9 * n_cols)]
        coords[color] = np.array([raw_start + crop_raws, crop_cols])
    return unaligned_rgb, coords

def find_relative_shift_pyramid(img_a, img_b):
    # Your code here
    a_to_b = np.array([0, 0])
    return a_to_b


def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):
    # Your code here
    r_to_g = np.array([0, 0])
    b_to_g = np.array([0, 0])
    return r_to_g, b_to_g


def create_aligned_image(
    channels,
    channel_coords,
    r_to_g,
    b_to_g,
):
    # Your code here
    aligned_img = None
    return aligned_img


def find_relative_shift_fourier(img_a, img_b):
    # Your code here
    a_to_b = np.array([0, 0])
    return a_to_b


if __name__ == "__main__":
    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)
