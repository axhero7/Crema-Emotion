from torch import nn
import torch
from torchsummary import summary

class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.5)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.5)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_1 = nn.Linear(128*9*5, 20)
        self.dropout1 = nn.Dropout(0.7)
        self.linear_2 = nn.Linear(20, 6)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x = x.view(-1, 128*9*5)
        x = self.linear_2(self.dropout1(self.linear_1(x)))
        return x
    
class OptimizedCremaNet(nn.Module):
    def __init__(self, trial=None, input_channels=1, num_classes=6):
        super().__init__()
        self.trial = trial
        self.features = self._create_conv_layers()
        self.classifier = self._create_classifier()

    def _create_conv_layers(self):
        layers = []
        in_channels = 1
        num_blocks = self.trial.suggest_int('num_blocks', 3, 6) if self.trial else 5
        
        for i in range(num_blocks):
            out_channels = self.trial.suggest_categorical(f'channels_{i}', [16, 32, 64, 128, 256]) if self.trial else 128
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(self.trial.suggest_float(f'dropout_conv_{i}', 0.1, 0.5)) if self.trial else nn.Dropout2d(0.3)
            ])
            in_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        return nn.Sequential(*layers)

    def _create_classifier(self):
        in_features = self._get_conv_output()
        hidden_size = self.trial.suggest_categorical('hidden_size', [64, 128]) if self.trial else 64
        num_layers = self.trial.suggest_int('num_fc_layers', 1, 3) if self.trial else 2
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.trial.suggest_float(f'dropout_fc_{i}', 0.2, 0.6) if self.trial else 0.5))
            in_features = hidden_size
            hidden_size = hidden_size // 2
        
        layers.append(nn.Linear(in_features, 6))
        return nn.Sequential(*layers)

    def _get_conv_output(self):
        with torch.no_grad():
            dummy = torch.ones(1, 1, 64, 501)  # Match input spectrogram shape
            return self.features(dummy).view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
