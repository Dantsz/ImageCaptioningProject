import torch
from torch import nn

class P2Encoder(nn.Module):
    '''Encoder for the second prototype.
    This time I'm trying to use a normal CNN to generate the embeddings.
    '''

    def __init__(self, input_channels: int, input_width: int, input_height: int, d_model: int):
        '''
        Args:
            intput_channels: The number of input channels (e.g., 3 for RGB images)
            input_width: The width of the input image
            input_height: The height of the input image
            d_model: The dimension of the model embeddings, which should be the same as the input embeddings of the decoder
        '''
        assert input_channels == 3, "Currently only RGB images are supported"
        assert input_width % 16 == 0, "Input width must be a multiple of 16"
        assert input_height % 16 == 0, "Input height must be a multiple of 16"
        super(P2Encoder, self).__init__()
        self.d_model = d_model
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * (input_width // 16) * (input_height // 16), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: The input tensor of shape (batch_size, input_channels, input_width, input_height)
        Returns:
            A tensor of shape (batch_size, d_model) containing the embeddings
        '''
        batch_size = x.size(0)
        x = self.pool(self.bn1(self.conv1_1(x)))
        x = self.pool(self.bn2(self.conv2_1(x)))
        x = self.pool(self.bn3(self.conv3_1(x)))
        x = self.pool(self.bn4(self.conv4_1(x)))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        return x