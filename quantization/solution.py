from typing import List, Tuple

import numpy as np
import torch


# Task 1 (1 point)
class QuantizationParameters:
    def __init__(
        self,
        scale: np.float64,
        zero_point: np.int32,
        q_min: np.int32,
        q_max: np.int32,
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.q_min = q_min
        self.q_max = q_max

    def __repr__(self):
        return f"scale: {self.scale}, zero_point: {self.zero_point}"


def compute_quantization_params(
    r_min: np.float64,
    r_max: np.float64,
    q_min: np.int32,
    q_max: np.int32,
) -> QuantizationParameters:
    # your code goes here \/
    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = np.round((r_max * q_min - r_min * q_max) / (r_max - r_min)).astype(np.int32)
    return QuantizationParameters(scale, zero_point, q_min, q_max)
    # your code goes here /\


# Task 2 (0.5 + 0.5 = 1 point)
def quantize(r: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    q = np.round(r / qp.scale + qp.zero_point)
    return np.clip(q, qp.q_min, qp.q_max).astype(np.int8)
    # your code goes here /\


def dequantize(q: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return (q - qp.zero_point) * qp.scale
    # your code goes here /\


# Task 3 (1 point)
class MinMaxObserver:
    def __init__(self):
        self.min = np.finfo(np.float64).max
        self.max = np.finfo(np.float64).min

    def __call__(self, x: torch.Tensor):
        # your code goes here \/
        self.min = np.min([self.min, np.min(x.detach().numpy())])
        self.max = np.max([self.max, np.max(x.detach().numpy())])
        # your code goes here /\


# Task 4 (1 + 1 = 2 points)
def quantize_weights_per_tensor(
    weights: np.ndarray,
) -> Tuple[np.array, QuantizationParameters]:
    # your code goes here \/
    r_max = max(abs(np.min(weights)), abs(np.max(weights)))
    r_min = -r_max
    q_min = -127
    q_max = 127
    qp = compute_quantization_params(r_min, r_max, q_min, q_max)
    return quantize(weights, qp), qp
    # your code goes here /\


def quantize_weights_per_channel(
    weights: np.ndarray,
) -> Tuple[np.array, List[QuantizationParameters]]:
    # your code goes here \/
    qws = []
    qps = []
    for i in range(weights.shape[0]):
        r_max = max(abs(np.min(weights[i])), abs(np.max(weights[i])))
        r_min = -r_max
        q_min = -127
        q_max = 127
        qp = compute_quantization_params(r_min, r_max, q_min, q_max)
        qps.append(qp)
        qws.append(quantize(weights[i], qp))
    return np.array(qws), qps
    # your code goes here /\


# Task 5 (1 point)
def quantize_bias(
    bias: np.float64,
    scale_w: np.float64,
    scale_x: np.float64,
) -> np.int32:
    # your code goes here \/
    return np.round(bias / (scale_w * scale_x)).astype(np.int32)
    # your code goes here /\


# Task 6 (2 points)
def quantize_multiplier(m: np.float64) -> [np.int32, np.int32]:
    # your code goes here \/
    n = 0
    while m < 0.5:
        m *= 2
        n += 1
    while m >= 1:
        m /= 2
        n -= 1
    return n, np.ceil(m * (2**31)).astype(np.int32)
    # your code goes here /\


# Task 7 (2 points)
def multiply_by_quantized_multiplier(
    accum: np.int32,
    n: np.int32,
    m0: np.int32,
) -> np.int32:
    # your code goes here \/
    res = np.multiply(accum, m0, dtype=np.int64)
    comma_id = 32 + 1 - n 
    first_after_comma = 1 & (res >> (64 - comma_id - 1))
    res = (res >> (64 - comma_id)).astype(np.int32) 
    if first_after_comma:
        res += 1
    return res
    # your code goes here /\
