import torch
import torch.nn as nn

NAME = "CRNN"


class CRNN(nn.Module):

    def __init__(self, input_shape, num_classes: int) -> None:
        super().__init__()

        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
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

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)            # (B, C', H', W')
            _, c2, h2, w2 = out.shape
            lstm_input_size = h2 * c2             # features per timestep
            seq_len = w2                          # timesteps

        self.permute = lambda x: x.permute(0, 3, 2, 1)  # (B,C',H',W') → (B,W',H',C')
        self.reshape = lambda x: x.contiguous().view(x.size(0), seq_len, lstm_input_size)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(512, num_classes)  # 256*2 → num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)      # (B,C',H',W')
        x = self.permute(x)       # (B,W',H',C')
        x = self.reshape(x)       # (B,W',H'*C')
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)       # output seq; keep last step
        x = x[:, -1, :]           # (B, 512)
        x = self.classifier(x)    # (B, num_classes)
        return x


def create_model(input_shape, config):
    return CRNN(input_shape, config["num_classes"])