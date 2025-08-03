import torch.nn as nn


class Layer(nn.Module):
    @staticmethod
    def forward(x):
        return x


class Input(Layer):
    def __init__(self):
        super().__init__()


class Dense(Layer):
    def __init__(self):
        super().__init__()


class Flatten(Layer):
    def __init__(self):
        super().__init__()


class Conv2D(Layer):
    def __init__(self):
        super().__init__()


class BatchNormalization(Layer):
    def __init__(self):
        super().__init__()


class MaxPooling2D(Layer):
    def __init__(self):
        super().__init__()


class Permute(Layer):
    def __init__(self):
        super().__init__()


class Reshape(Layer):
    def __init__(self):
        super().__init__()


class LSTM(Layer):
    def __init__(self):
        super().__init__()


class Bidirectional(Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
