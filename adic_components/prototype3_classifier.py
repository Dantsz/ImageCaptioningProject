from adic_components.prototype3 import P3Encoder
import torch
import torch.nn.functional as F
import torch.nn as nn

class P3Classifier(nn.Module):
    def __init__(self, input_channels: int, input_height: int, input_width: int, num_classes, p3_encoder: P3Encoder|None = None, dropout: float = 0.25):
        super(P3Classifier, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        if p3_encoder is None:
            self.p3_encoder = P3Encoder(input_channels, input_height, input_width, 768)
        else:
            self.p3_encoder = p3_encoder
        self.classifier_input = 768 * (input_height // self.p3_encoder.downsampling_factor) * (input_width // self.p3_encoder.downsampling_factor)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.p3_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    print('Testing P3Classifier')
    model = P3Classifier(3, 224, 224, 10)
    input_example = torch.randn(1, 3, 224, 224)
    assert model(input_example).shape == (1, 10), f"Output shape is incorrect: {model(input_example).shape}"
