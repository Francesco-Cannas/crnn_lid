import torch
import torch.nn as nn

NAME = "Topcoder_CRNN_Finetune"


class TopcoderCRNNFinetune(nn.Module):

    def __init__(self, input_shape, num_classes: int) -> None:
        super().__init__()
        c, h, w = input_shape  # (C, H, W)

        self.features = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1)),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1)),
        )

        for p in self.features.parameters():
            p.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)  # (1, C', H', W')
            _, c2, h2, _ = out.shape
            lstm_input = c2 * h2

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        x = self.features(x)  # (B, C', H', W')
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, W', C', H')
        x = x.view(x.size(0), x.size(1), -1)  # (B, W', C'*H')
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # (B, W', 1024)
        x = x[:, -1, :]
        return self.classifier(x)


def create_model(input_shape, config):
    return TopcoderCRNNFinetune(input_shape, config["num_classes"])
