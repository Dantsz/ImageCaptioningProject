import torch
import copy
from typing import Optional, Tuple, Union
from torch import nn
from transformers import GPT2Model, GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_attention_mask_for_sdpa
from adic_components.DyT import DyT
from loguru import logger
class P2EncoderGluer(nn.Module):
    '''
        Adjusts the output of the encoder to be compatible with the decoder.
    '''
    def __init__(self, encoder_token_dim: int, encoder_seq_length: int, decoder_token_dim: int, dropout: float = 0.1):
        super(P2EncoderGluer, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(encoder_token_dim, decoder_token_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(decoder_token_dim, decoder_token_dim*4),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(decoder_token_dim*4, decoder_token_dim),
                                )
        self.positional_encoding = nn.Parameter(torch.randn(1, encoder_seq_length, decoder_token_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: The input tensor of shape (batch_size, encoder_token_dim)
        Returns:
            A tensor of shape (batch_size, decoder_token_dim) containing the embeddings
        '''
        x = self.mlp(x)
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
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: The input tensor of shape (batch_size, input_channels, input_width, input_height)
        Returns:
            A tensor of shape (batch_size, d_model) containing the embeddings
        '''

        identity = self.bn_res1(self.identity1(x))
        x = self.pool(self.act(self.bn1(self.conv1_1(x))))
        x = x + identity

        identity = self.bn_res2(self.identity2(x))
        x = self.pool(self.act(self.bn2(self.conv2_1(x))))
        x = x + identity

        identity = self.bn_res3(self.identity3(x))
        x = self.pool(self.act(self.bn3(self.conv3_1(x))))
        x = x + identity

        identity = self.bn_res4(self.identity4(x))
        x = self.pool(self.act(self.bn4(self.conv4_1(x))))
        x = x + identity

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (batch_size, d_model, seq_length) -> (batch_size, seq_length, d_modela, this is how the input to cross attention should look like
        x = self.gluer(x)
        return x


class P2DecoderCrossAttention(nn.Module):
    '''
        The cross attention layer of the decoder allows the decoder to attend to the encoder output, that is add contextual about the image to the text.
    '''
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super(P2DecoderCrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,dropout=dropout, batch_first=True)

    def forward(self, decoder_self_attention_output: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x, _ = self.cross_attn(decoder_self_attention_output, encoder_output, encoder_output) # the second parameter returned is the attention weights, we don't need them, for now anyways, and hopefully never
        return x
class P2GPTBlock(GPT2Model):
    '''
        The GPT block of the decoder is very similar to a GPT-2, with addition of a cross attention layer and decoupled embedding/de-embedding MLPs,
        this is done to allow fine-tuning of the model without touching the self-attention weights.
    '''
    def __init__(self, config: GPT2Config):
        super(P2GPTBlock, self).__init__(config=config)
class P2Decoder(nn.Module):
    def __init__(self, gpt2_config: GPT2Config, lm_dropout: float = 0.3):
        super(P2Decoder, self).__init__()
        self.gpt2 = P2GPTBlock(gpt2_config)
        self.hidden_size = gpt2_config.n_embd
        self.vocab_size = gpt2_config.vocab_size
        self.cross_attention = P2DecoderCrossAttention(self.hidden_size, gpt2_config.n_head)# use the same number of heads as the GPT-2 model
        self.norm = DyT(self.hidden_size)
        # these are the embeddings that that the decoder outputs, the original GPT-2 model uses the same embeddings for input and output
        # but then we can't fine-tune the model without touching the self-attention weights
        # so we use a separate embedding layer for the output
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(lm_dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(self.gpt2.wte.weight.clone())

    def forward(self, x, encoder_output, attention_mask: Optional[torch.FloatTensor] = None):
        # batch_size, seq_length, d_model = gpt2_output.shape
        x_shape = x.shape
        encoder_output_shape = encoder_output.shape
        assert len(x_shape) == 2, f"Input shape mismatch: {x_shape}, format should be (batch_size, seq_length)"
        assert len(encoder_output_shape) == 3, f"Encoder output shape mismatch: {encoder_output_shape}, format should be (batch_size, seq_length, d_model)"
        assert x_shape[0] == encoder_output_shape[0], f"Batch size mismatch: {x_shape[0]} != {encoder_output_shape[0]}"
        logger.trace("Decoder input shape: {}", x.shape)
        logger.trace("Encoder output shape: {}", encoder_output.shape)
        x = self.gpt2(x, attention_mask=attention_mask).last_hidden_state # the output of the GPT-2 block is (batch_size, seq_length, d_model)
        logger.trace("Decoder output shape: {}", x.shape)
        x = self.cross_attention(x, encoder_output) + x
        x = self.norm(x)
        logger.trace("Cross attention output shape: {}", x.shape)
        x = self.norm(self.mlp(x)) + x # DyT and Norm
        x = self.lm_head(x)
        logger.trace("LM head output shape: {}", x.shape)
        return x

class P2ECDEC(nn.Module):
    def __init__(self, input_channels: int, input_width: int, input_height: int, d_model: int, decoder: nn.Module):
        super(P2ECDEC, self).__init__()
        self.encoder = P2Encoder(input_channels, input_width, input_height, d_model)
        self.decoder = decoder

    def forward(self, tokens: torch.Tensor, images: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        x = self.encoder(images)
        x = self.decoder(tokens, x, attention_mask=attention_mask)
        return x

    def generate(self, images: torch.Tensor, max_length: int = 16) -> torch.Tensor:
        batch_size = images.shape[0]
        assert batch_size == 1, "Batch size must be 1 for generation, currently"
        tokens = torch.ones(1, 1).long().to(images.device) * self.decoder.gpt2.config.bos_token_id
        while tokens.shape[1] < max_length:
            # get the last token and pass it to the decoder
            x = self.forward(tokens, images)
            # get the last token
            x = torch.argmax(x[:, -1, :], dim=1).unsqueeze(0)
            tokens = torch.cat([tokens, x], dim=1).contiguous()
            if tokens[0, -1] == self.decoder.gpt2.config.eos_token_id:
                break
        return tokens

if __name__ == "__main__":
    # Example usage
    encoder = P2Encoder(input_channels=3, input_width=224, input_height=224, d_model=768)
    # Create a random input tensor
    x = torch.randn(32, 3, 24, 224)  # (batch_size, channels, height, width)

    # Forward pass through the encoder
    encoded_x = encoder(x)
    print("Encoded shape:", encoded_x.shape)
