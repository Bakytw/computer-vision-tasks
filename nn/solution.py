from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia 
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return np.maximum(inputs, 0)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs * (self.forward_inputs >= 0)
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/
        exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exponents / np.sum(exponents, axis=1, keepdims=True)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        # your code here \/
        n, d = grad_outputs.shape
        grad_inputs = np.zeros((n, d))
        for i in range(n):
            grad_inputs[i] = grad_outputs[i] @ (np.diag(self.forward_outputs[i]) - np.outer(self.forward_outputs[i], self.forward_outputs[i]))
        return grad_inputs
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        return inputs @ self.weights.T + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        self.weights_grad = grad_outputs.T @ self.forward_inputs
        self.biases_grad = np.sum(grad_outputs, axis=0)
        return grad_outputs @ self.weights
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        # your code here \/
        n = y_gt.shape[0]
        y_pred = np.clip(y_pred, eps, 1.0)
        loss = -np.sum(y_gt * np.log(y_pred)) / n
        return np.array([loss])
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        # your code here \/
        grad = np.zeros_like(y_gt)
        grad[y_gt == 1] = -1 / (y_gt.shape[0] * np.clip(y_pred[y_gt == 1], eps, 1))
        return grad
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGD(1e-3))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(input_shape=(784, ), units=128))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())
    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, 32, 10)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # your code here \/
    n, d, ih, iw = inputs.shape
    c, d, kh, kw = kernels.shape
    oh = ih - kh + 1 + 2 * padding
    ow = iw - kw + 1 + 2 * padding
    inputs = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    n_stride, d_stride, h_stride, w_stride = inputs.strides
    inputs = np.lib.stride_tricks.as_strided(inputs, (n, d, oh, ow, kh, kw), (n_stride, d_stride, h_stride, w_stride, h_stride, w_stride))
    return np.einsum('ndhwkl,cdkl->nchw', inputs, kernels[:, :, ::-1, ::-1])
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        return convolve(inputs, self.kernels, (self.kernel_size - 1) // 2) + self.biases[np.newaxis, :, np.newaxis, np.newaxis]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        self.biases_grad = np.sum(grad_outputs, axis=(3, 2, 0))
        X = self.forward_inputs[:, :, ::-1, ::-1]
        self.kernels_grad = convolve(X.transpose(1, 0, 2, 3), grad_outputs.transpose(1, 0, 2, 3), (self.kernel_size - 1) // 2).transpose(1, 0, 2, 3)
        return convolve(grad_outputs, self.kernels[:, :, ::-1, ::-1].transpose(1, 0, 2, 3), (self.kernel_size - 1) // 2)
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        n, d, ih, iw = inputs.shape
        n_stride, d_stride, h_stride, w_stride = inputs.strides
        oh, ow = ih // self.pool_size, iw // self.pool_size
        inputs = np.lib.stride_tricks.as_strided(inputs, (n, d, oh, ow, self.pool_size, self.pool_size), (n_stride, d_stride, self.pool_size * h_stride, self.pool_size * w_stride, h_stride, w_stride))
        if self.pool_mode == 'avg':
            return np.mean(inputs, axis = (-1, -2))
        elif self.pool_mode == 'max':
            self.forward_idxs = np.argmax(inputs.reshape((-1, self.pool_size * self.pool_size)), axis=-1)
            return np.max(inputs, axis = (-1, -2))
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        n, d, oh, ow = grad_outputs.shape
        pool_shape = (n, d, oh, ow, self.pool_size, self.pool_size)
        if self.pool_mode == 'avg':
            inputs = np.ones_like(self.forward_inputs).reshape(*pool_shape)
            inputs = inputs * grad_outputs.reshape((n, d, oh, ow, 1, 1)) / (self.pool_size * self.pool_size)
        elif self.pool_mode == 'max':
            inputs = np.zeros_like(self.forward_inputs).reshape((-1, self.pool_size * self.pool_size))
            inputs[tuple(np.indices((len(self.forward_idxs),))[0]), self.forward_idxs] = grad_outputs.reshape(-1)
            inputs = inputs.reshape(*pool_shape)
        return inputs.transpose(0, 1, 2, 4, 3, 5).reshape((n, d, oh * self.pool_size, ow * self.pool_size))
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            mean = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
            var = np.var(inputs, axis=(0, 2, 3), keepdims=True)
            self.forward_inverse_std = 1 / np.sqrt(var + eps)
            self.forward_centered_inputs = inputs - mean
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * np.mean(inputs, axis=(0, 2, 3))
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * np.var(inputs, axis=(0, 2, 3))
            return self.gamma[:, np.newaxis, np.newaxis] * self.forward_normalized_inputs + self.beta[:, np.newaxis, np.newaxis]
        else:
            self.forward_inverse_std = 1 / np.sqrt(self.running_var[:, np.newaxis, np.newaxis] + eps)
            return self.gamma[:, np.newaxis, np.newaxis] * ((inputs - self.running_mean[:, np.newaxis, np.newaxis]) * self.forward_inverse_std) + self.beta[:, np.newaxis, np.newaxis]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        n, d, h, w = grad_outputs.shape

        grad_one = grad_outputs * self.gamma[:, np.newaxis, np.newaxis]
        grad_two = grad_one * self.forward_inverse_std
        grad_var = -0.5 * np.sum(grad_one * self.forward_centered_inputs, axis=(0, 2, 3), keepdims=True) * self.forward_inverse_std ** 3
        grad_mean = np.sum(-grad_two, axis=(0, 2, 3), keepdims=True) + grad_var * np.sum(-2 * self.forward_centered_inputs, axis=(0, 2, 3), keepdims=True) / (n * h * w)

        self.gamma_grad = np.sum(grad_outputs * self.forward_normalized_inputs, axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        
        return (2 * self.forward_centered_inputs * grad_var + grad_mean) / (n * h * w) + grad_two
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        # your code here \/
        self.forward_inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1) 
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        # your code here \/
        return grad_outputs.reshape(self.forward_inputs_shape) 
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask = np.random.rand(*inputs.shape) > self.p
            return inputs * self.forward_mask
        else:
            return inputs * (1 - self.p)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs * self.forward_mask
        # your code here /\

# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(1e-3, 0.99))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(output_channels = 16, kernel_size= 3, input_shape=(3, 32, 32)))
    model.add(ReLU())
    # model.add(Pooling2D())

    model.add(Conv2D(32, 3))
    model.add(ReLU())
    # model.add(Pooling2D())

    model.add(Conv2D(64, 3))
    model.add(ReLU())
    # model.add(Pooling2D())
    model.add(BatchNorm())

    model.add(Flatten())
    model.add(Dense(10))
    model.add(ReLU())
    model.add(Softmax())
    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, batch_size=32, epochs=10)

    # your code here /\
    return model


# ============================================================================
