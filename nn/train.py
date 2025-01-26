import numpy as np

inputs = np.array([[[[  20,  200,   -5,   23],
       [ -13,  134,  119,  100],
       [ 120,   32,   49,   25],
       [-120,   12,   9,   23]]]])
pool_size = 2
pool_mode = 'max'
output_shape = (1, 2, 2)
b, d, _, _ = inputs.shape
batch_stride, channel_stride, rows_stride, columns_stride = inputs.strides

new_shape = (b, *output_shape, pool_size, pool_size)
new_strides = (
    batch_stride,
    channel_stride,
    pool_size * rows_stride,
    pool_size * columns_stride,
    rows_stride,
    columns_stride)

input_windows = np.lib.stride_tricks.as_strided(inputs, new_shape, new_strides)
print(input_windows)
print(np.max(input_windows, axis = (-1, -2)))
print(inputs.reshape((-1, pool_size * pool_size)))
print(np.argmax(input_windows.reshape(b, *output_shape, pool_size ** 2), axis=-1, keepdims=True))

# if pool_mode == 'max':
#     max_ids = np.argmax(input_windows.reshape(b, *output_shape, pool_size ** 2), axis=-1, keepdims=True)
#     print(input_windows.max(axis=(-1, -2)))
# elif pool_mode == 'avg':
#     print(input_windows.mean(axis=(-1, -2)))
# print(np.max(a, axis = 0, keepdims=True))