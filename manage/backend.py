import torch


def epsilon():
    return torch.finfo(torch.float32).eps


def sum(x, axis=None):
    return torch.sum(x, dim=axis)


def round(x):
    return torch.round(x)


def cast(x, dtype):
    return x.type(dtype)


def square(x):
    return torch.square(x)


def sqrt(x):
    return torch.sqrt(x)


def mean(x, axis=None):
    return torch.mean(x, dim=axis)


def image_data_format():
    return 'channels_last'
