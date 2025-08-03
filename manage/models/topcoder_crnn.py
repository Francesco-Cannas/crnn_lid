import torch
import torch.nn as nn

NAME = "Topcoder_CRNN"

class TopcoderCRNN(nn.Module):

    def __init__(self, input_shape, num_classes: int) -> None:
        super().__init__()
        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 16,  kernel_size=7, padding=3),
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
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)             # (B,C,H,W)
            _, c2, h2, w2 = out.shape
            self.seq_len = w2
            self.lstm_input = h2 * c2

        self.lstm = nn.LSTM(
            input_size=self.lstm_input,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        x = self.features(x)                       # (B,C,H,W)
        x = x.permute(0, 3, 2, 1)                  # (B,W,H,C)
        x = x.contiguous().view(x.size(0), self.seq_len, self.lstm_input)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.classifier(x)


def create_model(input_shape, config):
    return TopcoderCRNN(input_shape, config["num_classes"])