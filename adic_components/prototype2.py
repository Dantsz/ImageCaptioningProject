from typing import Optional
import torch
from torch import nn
from transformers import GPT2Model
class P2EncoderGluer(nn.Module):
    '''
        Adjusts the output of the encoder to be compatible with the decoder.
    '''
    def __init__(self, encoder_token_dim: int, encoder_seq_length: int, decoder_token_dim: int):
        super(P2EncoderGluer, self).__init__()
        self.proj = nn.Linear(encoder_token_dim, decoder_token_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, encoder_seq_length, decoder_token_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: The input tensor of shape (batch_size, encoder_token_dim)
        Returns:
            A tensor of shape (batch_size, decoder_token_dim) containing the embeddings
        '''
        x = self.proj(x)
        x = x + self.positional_encoding
        return x

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
        # derive sequence length from input dimensions
        self.seq_length = (input_width // 16) * (input_height // 16)
        self.input_channels = input_channels
        self.input_width = input_width
        self.input_height = input_height
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

        self.gluer = P2EncoderGluer(512, self.seq_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: The input tensor of shape (batch_size, input_channels, input_width, input_height)
        Returns:
            A tensor of shape (batch_size, d_model) containing the embeddings
        '''

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

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (batch_size, d_model, seq_length) -> (batch_size, seq_length, d_modela, this is how the input to cross attention should look like
        x = self.gluer(x)
        return x


class P2GPTBlock(GPT2Model):
    '''
        The GPT block of the decoder is very similar to a GPT-2, with addition of a cross attention layer and decoupled embedding/de-embedding MLPs,
        this is done to allow fine-tuning of the model without touching the self-attention weights.
    '''
    def __init__(self, d_model: int):
        super(P2GPTBlock, self).__init__()

    def forward(self, input_ids: Optional[torch.LongTensor], **kwargs):
        '''
        Args:
            input_ids: The input tensor of shape (batch_size, seq_length)
        Returns:
            A tensor of shape (batch_size, seq_length, d_model) containing the embeddings
        '''
        output = super().forward(input_ids, **kwargs)
        return output
class P2Decoder(nn.Module):
    def __init__(self):
        super(P2Decoder, self).__init__()
    def forward(self, x):
        pass

class P2ECDEC(nn.Module):
    def __init__(self, input_channels: int, input_width: int, input_height: int, d_model: int, decoder: nn.Module):
        super(P2ECDEC, self).__init__()
        self.encoder = P2Encoder(input_channels, input_width, input_height, d_model)
        self.decoder = decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    # Example usage
    encoder = P2Encoder(input_channels=3, input_width=224, input_height=224, d_model=768)
    # Create a random input tensor
    x = torch.randn(32, 3, 24, 224)  # (batch_size, channels, height, width)

    # Forward pass through the encoder
    encoded_x = encoder(x)
    print("Encoded shape:", encoded_x.shape)
