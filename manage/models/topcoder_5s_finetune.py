import torch
import torch.nn as nn

NAME = "topcoder_5s_finetune"

class Topcoder5sFinetune(nn.Module):

    def __init__(self, input_shape, num_classes: int) -> None:
        super().__init__()
        c, _, _ = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Dropout(0.5),
        )

        with torch.no_grad():
            flat = self.features(torch.zeros(1, *input_shape)).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_model(input_shape, config):
    return Topcoder5sFinetune(input_shape, config["num_classes"])