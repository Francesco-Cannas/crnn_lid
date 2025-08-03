import torch


def to_categorical(y, num_classes):
    y_tensor = torch.tensor(y, dtype=torch.long)
    return torch.nn.functional.one_hot(y_tensor, num_classes).numpy()
