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
        assert input_channels == 3 or input_channels == 1, "Currently only RGB or Grayscale images are supported"
        assert input_width % 16 == 0, "Input width must be a multiple of 16"
        assert input_height % 16 == 0, "Input height must be a multiple of 16"
        super(P2Encoder, self).__init__()
        self.d_model = d_model
        # CNN layers
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Residual connections
        self.identity1 = nn.Conv2d(input_channels, 64, kernel_size=1, stride=2, padding=0)
        self.identity2 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        self.identity3 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        self.identity4 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)

        # Batch normalization layers for the residuals
        self.bn_res1 = nn.BatchNorm2d(64)
        self.bn_res2 = nn.BatchNorm2d(128)
        self.bn_res3 = nn.BatchNorm2d(256)
        self.bn_res4 = nn.BatchNorm2d(512)

        # Fully connected layer
        self.fc1 = nn.Linear(512 * (input_width // 16) * (input_height // 16), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: The input tensor of shape (batch_size, input_channels, input_width, input_height)
        Returns:
            A tensor of shape (batch_size, d_model) containing the embeddings
        '''
        batch_size = x.size(0)

        identity = self.bn_res1(self.identity1(x))
        x = self.pool(self.bn1(self.conv1_1(x)))
        x = x + identity

        identity = self.bn_res2(self.identity2(x))
        x = self.pool(self.bn2(self.conv2_1(x)))
        x = x + identity

        identity = self.bn_res3(self.identity3(x))
        x = self.pool(self.bn3(self.conv3_1(x)))
        x = x + identity

        identity = self.bn_res4(self.identity4(x))
        x = self.pool(self.bn4(self.conv4_1(x)))
        x = x + identity

        x = x.view(batch_size, -1)
        x = self.fc1(x)

class P2Decoder(nn.Module):
    def __init__(self):
        super(P2Decoder, self).__init__()
    def forward(self, x):
        pass